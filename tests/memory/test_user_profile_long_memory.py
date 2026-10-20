"""用户画像模型、SQLite 存储和长期记忆服务测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.memory.profile_candidate import ProfileCandidate
from agent.memory.profile_service import (
    DEFAULT_PROFILE_MODEL,
    OllamaProfileCandidateExtractor,
    ProfileCandidateExtractor,
    ProfileCandidateProposal,
    UserProfileManager,
)
from agent.memory.profile_store import ProfileConflictError, ProfileStore
from agent.memory.schema import MemoryEvidence
from agent.memory.user_profile import (
    ProfileActor,
    ProfileCandidateStatus,
    ProfileValidationError,
    validate_profile_update,
)


class FixedCandidateExtractor(ProfileCandidateExtractor):
    """测试提取器：直接返回准备好的候选，不连接真实 Ollama。"""

    def __init__(self, proposals: list[ProfileCandidateProposal]) -> None:
        self.proposals = proposals

    def extract(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> list[ProfileCandidateProposal]:
        return self.proposals


class FakeHttpResponse:
    """模拟 urllib 响应，避免测试连接真实 Ollama。"""

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def make_proposal(
    user_id: str,
    field_name: str,
    value,
    source_actor: ProfileActor,
    quote: str | None,
    needs_confirmation: bool,
    confidence: float = 0.95,
) -> ProfileCandidateProposal:
    """创建带来源证据的测试候选。"""
    candidate = ProfileCandidate(
        user_id=user_id,
        field=field_name,
        value=value,
        reason="测试候选",
        needs_user_confirmation=needs_confirmation,
        confidence=confidence,
        evidence=MemoryEvidence(
            session_id="session-1",
            message_ids=["message-1"],
            excerpt="测试原文",
        ),
    )
    return ProfileCandidateProposal(
        candidate=candidate,
        source_actor=source_actor,
        explicit_quote=quote,
    )


class UserProfilePolicyTests(unittest.TestCase):
    """验证画像字段权限不依赖大模型自觉。"""

    def test_user_can_update_nickname_but_model_cannot(self) -> None:
        value = validate_profile_update("nickname", "  阿辰  ", ProfileActor.USER)
        self.assertEqual(value, "阿辰")

        with self.assertRaises(ProfileValidationError):
            validate_profile_update("nickname", "阿辰", ProfileActor.MODEL)

    def test_only_system_can_update_onboarding_state(self) -> None:
        value = validate_profile_update(
            "onboarding_completed",
            True,
            ProfileActor.SYSTEM,
        )
        self.assertTrue(value)

        with self.assertRaises(ProfileValidationError):
            validate_profile_update(
                "onboarding_completed",
                True,
                ProfileActor.USER,
            )


class OllamaCandidateExtractorTests(unittest.TestCase):
    """验证 Ollama JSON 响应可以转换为独立候选对象。"""

    def test_ollama_json_response_creates_explicit_candidate(self) -> None:
        model_json = (
            '{"candidates":[{"field":"nickname","value":"阿辰",'
            '"reason":"用户明确修改昵称","source":"user_explicit",'
            '"confidence":0.98,"needs_confirmation":false,'
            '"evidence":"以后叫我阿辰"}]}'
        )
        ollama_json = '{"response":' + json.dumps(model_json) + "}"
        response = FakeHttpResponse(ollama_json.encode("utf-8"))
        extractor = OllamaProfileCandidateExtractor()

        with mock.patch(
            "agent.memory.profile_service.urllib_request.urlopen",
            return_value=response,
        ):
            proposals = extractor.extract(
                "user-1",
                "以后叫我阿辰",
                "session-1",
                "message-1",
            )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].candidate.field, "nickname")
        self.assertEqual(proposals[0].candidate.value, "阿辰")
        self.assertEqual(proposals[0].source_actor, ProfileActor.USER)
        self.assertEqual(proposals[0].explicit_quote, "以后叫我阿辰")


class ProfileStoreTests(unittest.TestCase):
    """验证画像、候选和修改日志在一个 SQLite 文件中正确持久化。"""

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_directory.name) / "profiles.sqlite3"
        self.store = ProfileStore(database_path)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_profile_update_increments_revision_and_writes_log(self) -> None:
        profile = self.store.create_profile("user-1")
        self.assertEqual(profile.revision, 0)

        updated = self.store.update_profile_field(
            "user-1",
            "nickname",
            "阿辰",
            ProfileActor.USER,
            "用户明确修改昵称",
            expected_revision=0,
        )

        self.assertEqual(updated.nickname, "阿辰")
        self.assertEqual(updated.revision, 1)
        logs = self.store.list_change_logs("user-1")
        self.assertEqual(len(logs), 1)
        self.assertIsNone(logs[0].old_value)
        self.assertEqual(logs[0].new_value, "阿辰")

        with self.assertRaises(ProfileConflictError):
            self.store.update_profile_field(
                "user-1",
                "name",
                "小明",
                ProfileActor.USER,
                "使用过期 revision 修改",
                expected_revision=0,
            )

    def test_model_cannot_directly_update_profile_store(self) -> None:
        with self.assertRaises(ProfileValidationError):
            self.store.update_profile_field(
                "user-1",
                "nickname",
                "模型猜测的昵称",
                ProfileActor.MODEL,
                "模型尝试直接写入",
            )


class UserProfileManagerTests(unittest.TestCase):
    """验证画像管理器对候选执行自动应用、等待确认和拒绝。"""

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_directory.name) / "profiles.sqlite3"
        self.store = ProfileStore(database_path)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_explicit_nickname_is_applied_automatically(self) -> None:
        proposal = make_proposal(
            "user-1",
            "nickname",
            "阿辰",
            ProfileActor.USER,
            "以后叫我阿辰",
            needs_confirmation=False,
        )
        service = UserProfileManager(
            self.store,
            FixedCandidateExtractor([proposal]),
        )

        results = service.process_user_message(
            "user-1",
            "以后叫我阿辰",
            "session-1",
            "message-1",
        )

        self.assertEqual(results[0].status, ProfileCandidateStatus.APPLIED)
        self.assertEqual(service.get_profile("user-1").nickname, "阿辰")
        stored = self.store.get_candidate(results[0].candidate_id)
        self.assertEqual(stored.status, ProfileCandidateStatus.APPLIED)
        self.assertIn("昵称：阿辰", service.get_profile_context("user-1"))

    def test_model_inference_waits_for_user_confirmation(self) -> None:
        proposal = make_proposal(
            "user-1",
            "identity",
            "性格内向",
            ProfileActor.MODEL,
            None,
            needs_confirmation=True,
        )
        service = UserProfileManager(
            self.store,
            FixedCandidateExtractor([proposal]),
        )

        results = service.process_user_message("user-1", "今天不太想说话")
        result = results[0]

        self.assertEqual(result.status, ProfileCandidateStatus.PENDING)
        self.assertIsNone(service.get_profile("user-1").identity)

        profile = service.confirm_candidate("user-1", result.candidate_id)
        self.assertEqual(profile.identity, "性格内向")
        stored = self.store.get_candidate(result.candidate_id)
        self.assertEqual(stored.status, ProfileCandidateStatus.APPLIED)

    def test_hallucinated_value_is_not_applied_automatically(self) -> None:
        proposal = make_proposal(
            "user-1",
            "nickname",
            "模型编造的昵称",
            ProfileActor.USER,
            "以后叫我",
            needs_confirmation=False,
        )
        service = UserProfileManager(
            self.store,
            FixedCandidateExtractor([proposal]),
        )

        results = service.process_user_message("user-1", "以后叫我阿辰")

        self.assertEqual(results[0].status, ProfileCandidateStatus.PENDING)
        self.assertIsNone(service.get_profile("user-1").nickname)

    def test_user_message_cannot_change_system_only_field(self) -> None:
        proposal = make_proposal(
            "user-1",
            "onboarding_completed",
            True,
            ProfileActor.USER,
            "我已经完成初始化",
            needs_confirmation=False,
        )
        service = UserProfileManager(
            self.store,
            FixedCandidateExtractor([proposal]),
        )

        results = service.process_user_message("user-1", "我已经完成初始化")

        self.assertEqual(results[0].status, ProfileCandidateStatus.REJECTED)
        self.assertFalse(service.get_profile("user-1").onboarding_completed)

    def test_default_candidate_model_name_is_qwen2_5_3b(self) -> None:
        self.assertEqual(DEFAULT_PROFILE_MODEL, "qwen2.5:3b")


class ProfileTriggerTests(unittest.TestCase):
    """画像候选提取触发词规则的单元测试。"""

    def test_trigger_matches_profile_statements(self) -> None:
        from agent.memory.profile_service import matches_profile_trigger

        for text in [
            "以后叫我阿辰",
            "别叫我小名",
            "我叫张伟",
            "我是做程序员的",
            "我是一名学生",
            "我比较喜欢用中文交流",
            "我的时区是东八区",
            "你可以叫我队长",
        ]:
            self.assertTrue(matches_profile_trigger(text), f"应命中触发词: {text}")

    def test_trigger_skips_plain_chit_chat(self) -> None:
        from agent.memory.profile_service import matches_profile_trigger

        for text in [
            "今天天气真不错",
            "我有点累了",
            "嗯好的，我知道了",
            "",
            "你觉得我该怎么做",
        ]:
            self.assertFalse(matches_profile_trigger(text), f"不应命中触发词: {text}")


if __name__ == "__main__":
    unittest.main()
