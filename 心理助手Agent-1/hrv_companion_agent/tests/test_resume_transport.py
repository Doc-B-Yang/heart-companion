"""Backpressure and recovery regressions; no real board, network or user data."""
import asyncio
import hashlib
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'硬件开发'/'esp32_firmware'))
from gateway import PacketQueue, record_received, post_packet
from deploy_firmware import FILES, load_restore, rollback


class TestPeakBufferMemory(unittest.TestCase):
    def test_three_samples_preserve_legacy_peak_and_interval_results(self):
        import math
        from hrv import HRVAnalyzer
        bounded, legacy = HRVAnalyzer(), HRVAnalyzer()
        legacy.wave_n = 200
        # Long signal, movement and a timestamp gap exercise interpolation
        # and segment resets. Compare every detection and each report.
        for n in range(24000):
            timestamp = n * 10 + (100 if n >= 12000 else 0)
            value = 60000 + 1800 * math.sin(n * 2 * math.pi / 80) + 180 * math.sin(n * 2 * math.pi / 17)
            moving = 8000 <= n < 8100
            self.assertEqual(bounded.add_sample(value, timestamp, moving),
                             legacy.add_sample(value, timestamp, moving))
            self.assertLessEqual(len(bounded.ir), 3)
            self.assertLessEqual(len(bounded.t), 3)
            if n % 100 == 99:
                self.assertEqual(bounded.metrics(), legacy.metrics())


class TestGatewayBackpressure(unittest.IsolatedAsyncioTestCase):
    async def test_raw_record_survives_api_failure(self):
        with tempfile.TemporaryDirectory() as root:
            args=SimpleNamespace(record=Path(root)/'capture.jsonl', ble=False,
                                 url='http://localhost', user='diagnostic')
            packet={'schema':2,'seq':1}
            record_received(args,packet)
            class Offline:
                async def post(self,*args,**kwargs): raise OSError('offline')
            await post_packet(Offline(),args,packet)
            rows=[json.loads(line) for line in args.record.read_text().splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['packet'],packet)
            self.assertEqual(rows[0]['phase'],'received')

    async def test_slow_sender_keeps_only_bounded_latest_packets(self):
        queue=PacketQueue(maxsize=3)
        for seq in range(20):queue.put({'seq':seq})
        self.assertEqual(queue.dropped,17)
        self.assertEqual([(await queue.get())['seq'] for _ in range(3)],[17,18,19])

    async def test_expired_packets_are_not_delivered_as_fresh(self):
        queue=PacketQueue()
        queue.put({'seq':1},now=1)
        queue.put({'seq':2},now=10)
        with patch('gateway.time.monotonic',return_value=11):
            self.assertEqual((await queue.get())['seq'],2)
        self.assertEqual(queue.expired,1)

    async def test_reception_continues_during_http_delay(self):
        queue=PacketQueue(maxsize=2)
        queue.put({'seq':1})
        first=await queue.get()
        async def receive():
            for seq in range(2,10):
                queue.put({'seq':seq})
                await asyncio.sleep(0)
        task=asyncio.create_task(receive())
        await task
        self.assertEqual(first['seq'],1)
        self.assertEqual((await queue.get())['seq'],8)
        self.assertEqual(queue.dropped,6)


class TestDeploymentRecovery(unittest.TestCase):
    def test_incomplete_or_corrupt_restore_is_rejected_before_board_access(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)
            manifest={name:{'existed':False} for name in FILES}
            manifest.pop('main.py')
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(KeyError):load_restore(folder)
            manifest['main.py']={'existed':True,'sha256':hashlib.sha256(b'old').hexdigest()}
            (folder/'main.py').write_bytes(b'corrupted')
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_restore(folder)

    def test_rollback_attempts_remaining_files_after_one_failure(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)
            (folder/'main.py').write_bytes(b'old-main')
            seen=[]
            def replace(_transport,name,data):
                seen.append((name,data))
                if name=='config.py':raise OSError('disconnected')
            with patch('deploy_firmware.replace_remote',side_effect=replace):
                with self.assertRaisesRegex(RuntimeError,'Rollback incomplete'):
                    rollback(None,folder,['config.py','main.py'])
            self.assertEqual(seen,[('config.py',None),('main.py',b'old-main')])

    def test_restore_preserves_exact_bytes_and_absent_files(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)
            content=b'\xef\xbb\xbf# original\r\n'
            manifest={name:{'existed':False} for name in FILES}
            manifest['main.py']={'existed':True,'sha256':hashlib.sha256(content).hexdigest()}
            (folder/'main.py').write_bytes(content)
            (folder/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(load_restore(folder),{'main.py':content})


class TestFixedReplUART(unittest.TestCase):
    def test_compact_ble_preserves_all_inference_fields_and_sample_times(self):
        from ble_transport import encode_packet, frames, PACKET_FIELDS
        from ble_receiver import FrameAssembler
        packet={key:None for key in PACKET_FIELDS}
        packet.update(schema=2,device_id='device',boot_id='boot',seq=1,
                      device_ms=100000,eda_samples=[[99000+i*50,1.23456] for i in range(20)],
                      errors={'ppg':0,'imu':0,'ads':0,'fifo':0,'clock':0},
                      legacy_diagnostic='x'*1000)
        compact=encode_packet(packet)
        assembler=FrameAssembler(); result=None
        for frame in frames(compact,1): result=assembler.add(frame)
        self.assertEqual(result,{key:packet.get(key) for key in PACKET_FIELDS})
        self.assertLess(len(compact),len(json.dumps(packet))*0.6)
        for frame in frames('{"hc":1,"v":[]}',2): result=assembler.add(frame)
        self.assertIsNone(result)

    def test_negotiated_mtu_is_fixed_per_message_and_reassembles(self):
        from ble_transport import BLETransport
        from ble_receiver import FrameAssembler
        sent=[]
        transport=BLETransport.__new__(BLETransport)
        transport.ble=SimpleNamespace(gatts_notify=lambda c,h,f: sent.append(f))
        transport.handle=1; transport.connections={1}; transport.mtus={1:247}
        transport.queue=b''; transport.msg_id=0; transport.index=0
        transport.dropped=transport.errors=0
        transport._advertise_pending=False; transport._last_send=0
        packet={'schema':2,'test':'中文'*500}
        self.assertTrue(transport.enqueue(json.dumps(packet,ensure_ascii=False)))
        transport._irq(21,(1,100))
        with patch('ble_transport.time.ticks_ms',return_value=1000,create=True), patch('ble_transport.time.ticks_diff',return_value=8,create=True):
            while transport.queue: transport.service()
        self.assertEqual(transport.payload_size,238)
        self.assertLess(len(sent),20)
        assembler=FrameAssembler(); result=None
        for frame in sent:
            self.assertLessEqual(len(frame),244)
            result=assembler.add(frame)
        self.assertEqual(result,packet)
        self.assertTrue(transport.enqueue('{}'))
        self.assertEqual(transport.payload_size,91)

    def test_ble_lazy_frames_reassemble_after_notify_retry(self):
        from ble_transport import BLETransport
        from ble_receiver import FrameAssembler
        sent=[]
        class Radio:
            fail=True
            def gatts_notify(self,conn,handle,frame):
                if self.fail:
                    self.fail=False
                    raise OSError('busy')
                sent.append(frame)
        transport=BLETransport.__new__(BLETransport)
        transport.ble=Radio();transport.handle=1;transport.connections={1}
        transport.queue=b'';transport.msg_id=0;transport.index=0
        transport.dropped=transport.errors=0
        transport._advertise_pending=False;transport._last_send=0
        packet={'schema':2,'test':'x'*1000}
        self.assertTrue(transport.enqueue(json.dumps(packet)))
        self.assertFalse(transport.enqueue('{}'))
        with patch('ble_transport.time.ticks_ms',return_value=1000,create=True), patch('ble_transport.time.ticks_diff',return_value=8,create=True):
            while transport.queue: transport.service()
        assembler=FrameAssembler();result=None
        for frame in sent:
            self.assertLessEqual(len(frame),20)
            result=assembler.add(frame)
        self.assertEqual(result,packet)
        self.assertEqual(transport.errors,1)

    def test_ble_waits_for_pacing_and_retries_identical_fragment(self):
        from ble_transport import BLETransport
        attempts=[]
        class Radio:
            def gatts_notify(self,conn,handle,frame):
                attempts.append(frame)
                if len(attempts)==1: raise OSError(12)
        transport=BLETransport.__new__(BLETransport)
        transport.ble=Radio(); transport.handle=1; transport.connections={1}
        transport.mtus={1:23}; transport.queue=b''; transport.msg_id=0
        transport.index=0; transport.dropped=transport.errors=0
        transport._advertise_pending=False; transport._last_send=0
        self.assertTrue(transport.enqueue('x'*60))
        with patch('ble_transport.time.ticks_ms',return_value=7,create=True), patch('ble_transport.time.ticks_diff',side_effect=lambda a,b:a-b,create=True):
            transport.service()
        self.assertEqual(attempts,[])
        with patch('ble_transport.time.ticks_ms',return_value=8,create=True), patch('ble_transport.time.ticks_diff',side_effect=lambda a,b:a-b,create=True):
            transport.service()
        self.assertEqual(transport.index,0)
        self.assertEqual(transport.errors,1)
        with patch('ble_transport.time.ticks_ms',return_value=16,create=True), patch('ble_transport.time.ticks_diff',side_effect=lambda a,b:a-b,create=True):
            transport.service()
        self.assertEqual(attempts[0],attempts[1])
        self.assertEqual(transport.index,1)
        self.assertEqual(len(attempts),2)

    def test_bounded_chunks_partial_write_and_busy_queue(self):
        from serial_transport import USBTransport
        written=[];sizes=[]
        class UART:
            def write(self,data):
                sizes.append(len(data))
                count=min(len(data),7)
                written.append(data[:count])
                return count
        uart=UART()
        transport=USBTransport(stream=uart)
        self.assertTrue(transport.chunked)
        payload='x'*100
        self.assertTrue(transport.send(payload))
        self.assertFalse(transport.send('new'))
        while transport.pending:transport.service()
        self.assertEqual(''.join(written),payload+'\n')
        self.assertLessEqual(max(sizes),32)
        self.assertEqual(transport.dropped,1)
