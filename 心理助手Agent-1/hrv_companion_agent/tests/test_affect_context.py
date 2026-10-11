"""Subjective VA contracts; fake AI and temporary stores, never participant data."""
import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

from hrv_companion_agent import config
from hrv_companion_agent.backend import routes
from hrv_companion_agent.backend.main import app
from hrv_companion_agent.core.affect_context import (
    AffectContext, TTL_SECONDS, infer_language_valence, prompt_context)
from hrv_companion_agent.core.label_store import LabelStore, SelfReportStore
from hrv_companion_agent.core.llm_client import CompanionLLM


class FakeLanguageLLM:
    """Only structured inference is fake; chat can remain the offline implementation."""
    online = True

    def __init__(self, response=None, error=None):
        self.response = response or '{"valence":0.75,"confidence":0.9,"current_self":true}'
        self.error = error
        self.messages = []

    def invoke_valence(self, messages):
        self.messages.append(messages)
        if self.error:
            raise self.error
        return self.response


class TestAffectContext(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.context = AffectContext(clock=lambda: self.now)

    def language(self, value=0.75, confidence=0.9):
        token = self.context.begin_language()
        return self.context.finish_language(token, {
            'value': value, 'confidence': confidence, 'status': 'available',
            'input_source': 'text', 'reason': 'synthetic test estimate'})

    def test_report_axes_normalized_without_fabricated_dominance(self):
        for valence, arousal in ((1, 9), (5, 5), (9, 1)):
            with self.subTest(valence=valence):
                state = self.context.record_report(valence, arousal)
                self.assertEqual(state['valence']['value'], (valence-5)/4)
                self.assertEqual(state['arousal']['value'], (arousal-5)/4)
                self.assertEqual(state['valence']['score_1_9'], valence)
                self.assertEqual(state['valence']['source'], 'self_report')
                self.assertFalse(state['valence']['inferred'])
                self.assertEqual(state['valence']['scope'], 'preceding_60_second_window')
                self.assertTrue(state['coordinates']['complete'])
                self.assertIsNone(state['dominance'])
                self.assertFalse(state['validated'])

    def test_independent_report_priority_conflict_and_omitted_axes(self):
        self.language(1.0)
        state = self.context.record_report(1, 8)
        self.assertEqual(state['coordinates'], {'valence': -1.0, 'arousal': 0.75, 'complete': True})
        self.assertTrue(state['conflict'])
        self.assertEqual(state['language_estimate']['value'], 1.0)
        self.assertIn('尊重自评', prompt_context(state))
        # An omitted new arousal rating cannot inherit the old one.
        state = self.context.record_report(5)
        self.assertIsNone(state['arousal']['value'])
        self.assertEqual(state['valence']['value'], 0)
        fresh = AffectContext(clock=lambda: self.now)
        fresh.record_report(9, 1)
        state = fresh.record_report()
        self.assertIsNone(state['valence']['value'])
        self.assertIsNone(state['arousal']['value'])

    def test_report_and_language_expire_at_boundary(self):
        self.context.record_report(3, 7)
        self.language(-0.5)
        self.now += TTL_SECONDS-0.1
        self.assertTrue(self.context.snapshot()['coordinates']['complete'])
        self.now += 0.1
        state = self.context.snapshot()
        self.assertIsNone(state['valence']['value'])
        self.assertIsNone(state['arousal']['value'])
        self.assertEqual(state['language_estimate']['status'], 'unknown')
        self.assertFalse(state['conflict'])
        self.assertEqual(prompt_context(state), '')

    def test_language_has_no_arousal_and_snapshot_is_copy(self):
        state = self.language()
        self.assertEqual(state['valence']['source'], 'text_inference')
        self.assertTrue(state['valence']['inferred'])
        self.assertIsNone(state['arousal']['value'])
        self.assertFalse(state['coordinates']['complete'])
        state['language_estimate']['value'] = -1
        self.assertEqual(self.context.snapshot()['valence']['value'], 0.75)
        context_prompt = prompt_context(self.context.snapshot())
        self.assertIn('未经验证推测', context_prompt)
        self.assertNotIn('synthetic test estimate', context_prompt)

    def test_purge_and_newer_requests_block_inflight_results(self):
        stale = self.context.begin_language()
        self.context.clear()
        estimate = {'value': 0.5, 'confidence': 0.8, 'status': 'available', 'input_source': 'text'}
        self.context.finish_language(stale, estimate)
        self.assertIsNone(self.context.snapshot()['valence']['value'])
        old = self.context.begin_language()
        current = self.context.begin_language()
        self.context.finish_language(current, {**estimate, 'value': -0.5})
        self.context.finish_language(old, estimate)
        self.assertEqual(self.context.snapshot()['valence']['value'], -0.5)

    def test_report_rejects_invalid_scores_without_mutation(self):
        self.context.record_report(6, 4)
        for invalid in (True, '6', 6.0, 0, 10, float('nan')):
            with self.subTest(score=invalid):
                with self.assertRaises(ValueError):
                    self.context.record_report(invalid, 5)
                self.assertEqual(self.context.snapshot()['valence']['score_1_9'], 6)


class TestLanguageValence(unittest.TestCase):
    def test_known_current_self_with_transcript_provenance(self):
        fake = FakeLanguageLLM()
        result = infer_language_valence(fake, '现在我感觉很开心', 'speech_transcript')
        self.assertEqual(result['value'], 0.75)
        self.assertEqual(result['status'], 'available')
        self.assertEqual(result['input_source'], 'speech_transcript')
        self.assertEqual(json.loads(fake.messages[0][1]['content']), {'utterance': '现在我感觉很开心'})
        self.assertIn('不推断唤醒度', fake.messages[0][0]['content'])

    def test_malformed_numeric_ambiguous_responses_fail_closed(self):
        responses = (
            'not JSON', '```json\n{"valence":0.8}\n```', '[]', 'null',
            '{"valence":2,"confidence":0.9,"current_self":true}',
            '{"valence":true,"confidence":0.9,"current_self":true}',
            '{"valence":NaN,"confidence":0.9,"current_self":true}',
            '{"valence":0.8,"confidence":Infinity,"current_self":true}',
            '{"valence":0.8,"confidence":true,"current_self":true}',
            '{"valence":0.8,"confidence":1.1,"current_self":true}',
            '{"valence":0.8,"confidence":0.59,"current_self":true}',
            '{"valence":0.8,"confidence":0.9,"current_self":false}',
            '{"valence":0.8,"confidence":0.9,"current_self":"true"}',
            '{"valence":null,"confidence":0.9,"current_self":false}',
            '{"confidence":0.9,"current_self":true}',
            'x'*4097,
        )
        for response in responses:
            with self.subTest(response=response[:80]):
                result = infer_language_valence(FakeLanguageLLM(response), 'synthetic test expression')
                self.assertEqual(result['status'], 'unknown')
                self.assertIsNone(result['value'])

    def test_offline_empty_and_provider_error_do_not_generate_estimates(self):
        fake = FakeLanguageLLM()
        fake.online = False
        self.assertEqual(infer_language_valence(fake, '感觉不错')['status'], 'unknown')
        self.assertEqual(fake.messages, [])
        fake.online = True
        self.assertEqual(infer_language_valence(fake, '   ')['status'], 'unknown')
        self.assertEqual(fake.messages, [])
        result = infer_language_valence(FakeLanguageLLM(error=RuntimeError('private-provider-detail')), '感觉不错')
        self.assertEqual(result['status'], 'unknown')
        self.assertNotIn('private-provider-detail', json.dumps(result))

    def test_structured_invocation_is_bounded_and_separate_from_chat(self):
        llm = CompanionLLM.__new__(CompanionLLM)
        llm._backend = 'openai'
        llm.model = 'synthetic-test-model'
        llm._overrides = {'temperature': 1.8, 'top_p': 0.3}
        llm._last_params = {'temperature': 0.7}
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
        client = Mock()
        client.with_options.return_value = client
        client.chat.completions.create.return_value = response
        llm._client = client
        self.assertEqual(llm.invoke_valence([{'role': 'user', 'content': 'synthetic'}]), '{}')
        client.with_options.assert_called_once_with(timeout=12.0, max_retries=0)
        options = client.chat.completions.create.call_args.kwargs
        self.assertEqual(options['temperature'], 0)
        self.assertEqual(options['max_tokens'], 180)
        self.assertNotIn('top_p', options)
        self.assertFalse(options['stream'])
        self.assertEqual(llm._last_params, {'temperature': 0.7})
        client.chat.completions.create.side_effect = RuntimeError('synthetic failure')
        with self.assertRaises(RuntimeError):
            llm.invoke_valence([])
        self.assertEqual(client.chat.completions.create.call_count, 2)
        llm._backend = 'offline'
        with self.assertRaises(RuntimeError):
            llm.invoke_valence([])
        self.assertEqual(client.chat.completions.create.call_count, 2)


class TestAffectAPI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='affect-api-test-')
        self.config_patches = []
        for key, value in {
            'DATA_DIR': Path(self.temp.name),
            'MEMORY_DB_PATH': Path(self.temp.name)/'memory.db',
            'LLM_API_KEY': '', 'TTS_ENABLED': False, 'MULTIMODAL_MODEL_ENABLED': False,
            'MEMORY_FTA_ENABLED': False,
        }.items():
            item = patch.object(config, key, value)
            item.start()
            self.config_patches.append(item)
        self.old_systems = routes._systems
        self.old_labels = routes._labels
        self.old_reports = routes._experiment_reports
        routes._systems = {}
        routes._labels = LabelStore(Path(self.temp.name)/'labels.jsonl')
        routes._experiment_reports = SelfReportStore(Path(self.temp.name)/'reports.jsonl')
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        for system in routes._systems.values():
            system.close()
        routes._systems = self.old_systems
        routes._labels = self.old_labels
        routes._experiment_reports = self.old_reports
        for item in reversed(self.config_patches):
            item.stop()
        self.temp.cleanup()

    def post_report(self, **extra):
        return self.client.post('/api/dataset/self-report', json={
            'user_id': 'affect-test', 'label': '平静', 'note': 'synthetic test report', **extra})

    def test_no_sensor_state_report_axes_and_per_user_isolation(self):
        response = self.post_report(valence_score=9, arousal_score=2)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['training_sample_saved'])
        state = self.client.get('/api/emotion/state', params={'user_id': 'affect-test'}).json()
        self.assertIsNone(state['emotion'])
        self.assertEqual(state['affect_context']['coordinates'], {'valence': 1, 'arousal': -0.75, 'complete': True})
        self.assertEqual(routes._labels.export('affect-test'), [])
        records = routes._experiment_reports.export('affect-test')
        self.assertEqual(records[0]['valence_1_9'], 9)
        other = self.client.get('/api/emotion/state', params={'user_id': 'other-test'}).json()
        self.assertIsNone(other['affect_context']['valence']['value'])
        response = self.post_report()
        self.assertIsNone(response.json()['affect_context']['valence']['value'])
        self.assertIsNone(response.json()['affect_context']['arousal']['value'])

    def test_inference_never_writes_independent_labels_or_arousal(self):
        system = routes.get_system('affect-test')
        with patch.object(system.llm, 'invoke_valence', return_value=FakeLanguageLLM().response), \
             patch.object(CompanionLLM, 'online', new_callable=lambda: property(lambda _: True)):
            response = self.client.post('/api/emotion/valence', json={
                'user_id': 'affect-test', 'text': '我现在很开心', 'input_source': 'speech_transcript'})
        self.assertEqual(response.status_code, 200, response.text)
        context = response.json()['affect_context']
        self.assertEqual(context['valence']['source'], 'text_inference')
        self.assertEqual(context['language_estimate']['input_source'], 'speech_transcript')
        self.assertIsNone(context['arousal']['value'])
        self.assertFalse(context['coordinates']['complete'])
        self.assertEqual(routes._labels.export('affect-test'), [])
        self.assertEqual(routes._experiment_reports.export('affect-test'), [])

    def test_chat_automatic_inference_and_bounded_prompt_context(self):
        system = routes.get_system('affect-test')
        with patch.object(system.llm, 'invoke_valence', return_value=FakeLanguageLLM().response), \
             patch.object(CompanionLLM, 'online', new_callable=lambda: property(lambda _: True)), \
             patch.object(system.llm, 'invoke', return_value='synthetic companion response') as invoke:
            response = self.client.post('/api/chat/message', json={
                'user_id': 'affect-test', 'message': '我现在很开心', 'input_source': 'speech_transcript'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['affect_context']['valence']['source'], 'text_inference')
        self.assertEqual(response.json()['affect_context']['language_estimate']['input_source'], 'speech_transcript')
        system_prompt = invoke.call_args.args[0][0]['content']
        self.assertIn('[主观情绪上下文]', system_prompt)
        self.assertIn('效价=0.75', system_prompt)
        self.assertIn('不是硬件识别', system_prompt)
        self.assertEqual(routes._labels.export('affect-test'), [])
        self.assertEqual(routes._experiment_reports.export('affect-test'), [])

    def test_websocket_validates_input_retains_connection_and_voice_source(self):
        system = routes.get_system('affect-test')
        with patch.object(system.llm, 'invoke_valence', return_value=FakeLanguageLLM().response) as infer, \
             patch.object(CompanionLLM, 'online', new_callable=lambda: property(lambda _: True)), \
             patch.object(system.agent, 'stream_run', side_effect=lambda _: iter(['first', 'second'])):
            with self.client.websocket_connect('/api/chat/stream') as ws:
                ws.send_text('{invalid json')
                self.assertEqual(ws.receive_json()['type'], 'error')
                for payload in (
                    [], None, {'message': ''}, {'message': True},
                    {'message': 'x'*5001}, {'message': '表达', 'user_id': ''},
                    {'message': '表达', 'user_id': 'x'*65},
                    {'message': '表达', 'input_source': 'audio_blob'},
                ):
                    with self.subTest(payload=str(payload)[:80]):
                        ws.send_json(payload)
                        self.assertEqual(ws.receive_json()['type'], 'error')
                self.assertEqual(infer.call_count, 0)
                ws.send_json({'user_id': 'affect-test', 'message': '我现在很开心',
                              'input_source': 'speech_transcript'})
                self.assertEqual(ws.receive_json()['type'], 'state')
                self.assertEqual(ws.receive_json(), {'type': 'chunk', 'content': 'first'})
                self.assertEqual(ws.receive_json(), {'type': 'chunk', 'content': 'second'})
                done = ws.receive_json()
                self.assertEqual(done['type'], 'done')
                self.assertEqual(done['affect_context']['language_estimate']['input_source'],
                                 'speech_transcript')
                # Normal generator exhaustion must not break the socket or Future.
                ws.send_json({'message': '我现在很开心'})
                self.assertEqual(ws.receive_json()['type'], 'state')
                self.assertEqual(ws.receive_json()['type'], 'chunk')
                self.assertEqual(ws.receive_json()['type'], 'chunk')
                done = ws.receive_json()
                self.assertEqual(done['type'], 'done')
                self.assertEqual(done['affect_context']['language_estimate']['input_source'], 'text')
                self.assertEqual(infer.call_count, 2)
        self.assertEqual(routes._labels.export('affect-test'), [])
        self.assertEqual(routes._experiment_reports.export('affect-test'), [])

    def test_purge_blocks_inflight_estimate_and_preserves_other_user(self):
        self.post_report(valence_score=8, arousal_score=3)
        other = routes.get_system('other-test')
        other.record_affect_report(2, 7)
        system = routes.get_system('affect-test')
        in_flight = system.subjective_affect.begin_language()
        response = self.client.post('/api/memory/purge', json={'user_id': 'affect-test', 'keep_profile': False})
        self.assertEqual(response.status_code, 200, response.text)
        system.subjective_affect.finish_language(in_flight, {
            'status': 'available', 'value': 1, 'confidence': 1, 'input_source': 'text'})
        self.assertIsNone(system.get_affect_context()['valence']['value'])
        self.assertEqual(routes._experiment_reports.export('affect-test'), [])
        self.assertEqual(other.get_affect_context()['valence']['score_1_9'], 2)

    def test_api_invalid_scores_and_input_contract_fail_without_records(self):
        for key in ('valence_score', 'arousal_score'):
            for invalid in (True, '8', 0, 10, 2.5):
                with self.subTest(key=key, value=invalid):
                    self.assertEqual(self.post_report(**{key: invalid}).status_code, 422)
        for payload in ({'text': ''}, {'text': 'x'*5001}, {'text': '表达', 'input_source': 'audio_blob'}, {'text': True}):
            with self.subTest(payload=list(payload)):
                response = self.client.post('/api/emotion/valence', json={'user_id': 'affect-test', **payload})
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(routes._labels.export('affect-test'), [])
        self.assertEqual(routes._experiment_reports.export('affect-test'), [])


class TestAffectWebSocketScheduling(unittest.TestCase):
    def test_initial_inference_and_later_resumes_leave_event_loop_responsive(self):
        started = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        worker_ids, released_while_waiting, sent = [], [], []

        def blocking_stream(message, input_source='text'):
            self.assertEqual(input_source, 'speech_transcript')
            for index, chunk in enumerate(('first', 'second')):
                worker_ids.append(threading.get_ident())
                started[index].set()
                # A coroutine on the ASGI event loop must release this wait.
                released_while_waiting.append(release[index].wait(timeout=1))
                yield chunk

        class FakeWebSocket:
            received = False

            async def accept(self):
                pass

            async def receive_json(self):
                if self.received:
                    raise WebSocketDisconnect()
                self.received = True
                return {'message': 'synthetic expression', 'input_source': 'speech_transcript'}

            async def send_json(self, payload):
                sent.append(payload)

        system = SimpleNamespace(get_state=lambda: None, stream_chat=blocking_stream,
                                 get_affect_context=lambda: {'testing': True})

        async def scenario():
            event_loop_thread = threading.get_ident()

            async def release_from_event_loop():
                for index in range(2):
                    while not started[index].is_set():
                        await asyncio.sleep(0.001)
                    release[index].set()

            with patch.object(routes, 'get_system', return_value=system):
                await asyncio.wait_for(asyncio.gather(
                    routes.chat_stream(FakeWebSocket()), release_from_event_loop()), timeout=4)
            return event_loop_thread

        event_loop_thread = asyncio.run(scenario())
        self.assertEqual(released_while_waiting, [True, True])
        self.assertTrue(all(worker != event_loop_thread for worker in worker_ids))
        self.assertEqual([payload['type'] for payload in sent], ['state', 'chunk', 'chunk', 'done'])


if __name__ == '__main__':
    unittest.main()
