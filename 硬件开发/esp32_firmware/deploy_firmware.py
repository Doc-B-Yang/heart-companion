"""Stage/compile/backup current MicroPython files before replacing main last.

No erase or reflash of the MicroPython runtime. The backup is recoverable on PC;
--restore <backup directory> restores only filenames in the production allowlist.
"""
import argparse
import hashlib
import json
import time
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime'))
from mpremote.transport_serial import SerialTransport

FILES=['config.py','max30102.py','mpu6050.py','ads1115.py','hrv.py',
       'ble_transport.py','serial_transport.py','acquisition.py','main.py']
HERE=Path(__file__).resolve().parent


def read_remote(transport,name):
    size=int(transport.exec("import os; print(os.stat(%r)[6])"%name).strip())
    blocks=[]
    for offset in range(0,size,512):
        raw=transport.exec("import ubinascii,gc; gc.collect(); f=open(%r,'rb'); f.seek(%d); b=f.read(512); f.close(); print(ubinascii.hexlify(b).decode()); del b; gc.collect()"%(name,offset))
        blocks.append(bytes.fromhex(raw.decode().strip()))
    return b''.join(blocks)


def write_remote(transport,name,data):
    transport.exec("open(%r,'wb').close()"%name)
    for offset in range(0,len(data),512):
        block=data[offset:offset+512].hex()
        transport.exec("import ubinascii; f=open(%r,'ab'); f.write(ubinascii.unhexlify(%r)); f.close()"%(name,block))


def load_restore(directory):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    data={}
    for name in FILES:
        entry=manifest[name]  # An incomplete manifest must never delete files.
        if not isinstance(entry.get('existed'),bool):
            raise ValueError('Invalid backup manifest: '+name)
        if entry['existed']:
            content=(directory/name).read_bytes()
            if hashlib.sha256(content).hexdigest()!=entry['sha256']:
                raise ValueError('Restore backup checksum mismatch: '+name)
            data[name]=content
    return data


def replace_remote(transport,name,data):
    if data is None:
        if transport.fs_exists(name):
            transport.exec("import os; os.remove(%r)"%name)
        if transport.fs_exists(name):
            raise RuntimeError('Removal verification failed: '+name)
    else:
        write_remote(transport,name,data)
        if read_remote(transport,name)!=data:
            raise RuntimeError('Write verification failed: '+name)


def rollback(transport,backup,committed):
    failures=[]
    for name in committed:
        try:
            replace_remote(transport,name,(backup/name).read_bytes() if (backup/name).exists() else None)
        except Exception as exc:
            failures.append(name+': '+str(exc))
    if failures:
        raise RuntimeError('Rollback incomplete; keep board stopped and restore '+str(backup)+': '+'; '.join(failures))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port',default='COM6')
    ap.add_argument('--inspect',action='store_true')
    ap.add_argument('--restore',type=Path)
    ap.add_argument('--only',nargs='+',choices=FILES,help='Deploy only these production files, retaining a full backup')
    args=ap.parse_args()
    if args.only and (args.inspect or args.restore):
        ap.error('--only cannot be combined with inspect or restore')
    selected=[name for name in FILES if args.only is None or name in args.only]
    restore_data=load_restore(args.restore) if args.restore else None
    source_data={}
    if not args.inspect and not args.restore:
        for name in selected:
            data=(HERE/name).read_bytes()
            if data.startswith(b'\xef\xbb\xbf'):data=data[3:]
            compile(data,name,'exec')
            source_data[name]=data
    transport=SerialTransport(args.port, timeout=5)
    transport.serial.write_timeout=5
    print('PORT_OPEN',flush=True)
    # Classic raw REPL avoids an unbounded raw-paste handshake read on boards
    # that lose the acknowledgement after interrupting a busy sampler.
    transport.use_raw_paste=False
    entered=False
    committed=[]
    safe_to_restart=True
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    backup=HERE/'diagnostics'/'deployment-2026-10-04'/('board-backup-'+stamp)
    try:
        time.sleep(5)
        transport.serial.write(b'\r\x03\x03\x02')
        time.sleep(1)
        transport.serial.read(transport.serial.in_waiting)
        transport.enter_raw_repl(soft_reset=False);entered=True
        # Reset only after an acknowledged raw prompt, releasing sampler heap.
        transport.enter_raw_repl(soft_reset=True)
        info=transport.exec("import os,sys,gc,json; import bluetooth; print(json.dumps({'uname':list(os.uname()),'version':sys.version,'free_heap':gc.mem_free(),'ble_available':hasattr(bluetooth,'BLE')}))")
        print(info.decode().strip(),flush=True)
        if args.inspect:return
        backup.mkdir(parents=True)
        manifest={}
        for name in FILES:
            exists=transport.fs_exists(name)
            if exists:
                data=read_remote(transport,name)
                (backup/name).write_bytes(data)
                manifest[name]={'existed':True,'sha256':hashlib.sha256(data).hexdigest()}
            else:manifest[name]={'existed':False}
        (backup/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (backup/'board-info.json').write_bytes(info)
        print('BOARD_BACKUP',backup,flush=True)
        if args.restore:
            for name in FILES:
                committed.append(name)
                replace_remote(transport,name,restore_data.get(name))
            print('RESTORED',args.restore,flush=True)
            return
        for name in selected:
            staged='.__new_'+name
            data=source_data[name]
            write_remote(transport,staged,data)
            transport.exec("compile(open(%r).read(),%r,'exec')"%(staged,name))
            remote=read_remote(transport,staged)
            if remote!=data:raise RuntimeError('Staged verification failed: '+name)
            print('STAGED_VERIFIED',name,flush=True)
        for name in selected:
            committed.append(name)
            transport.exec("import os; os.rename(%r,%r)"%('.__new_'+name,name))
            if read_remote(transport,name)!=source_data[name]:
                raise RuntimeError('Committed verification failed: '+name)
        print('DEPLOYED',json.dumps(selected),flush=True)
    except Exception:
        if entered and committed:
            safe_to_restart=False
            rollback(transport,backup,committed)
            safe_to_restart=True
            print('ROLLED_BACK',backup,flush=True)
        raise
    finally:
        try:
            if entered:
                transport.exit_raw_repl()
                if safe_to_restart:transport.serial.write(b'\x04')
        finally:
            transport.close()

if __name__=='__main__':main()
