"""Course summary generation, keyword resolution and profile adjustments."""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from src.summarizer import (
    DEFAULT_KEYWORDS,
    _parse_summary_answer,
    apply_keyword_adjustments,
    build_summary_digest,
    build_summary_markdown,
    generate_course_summary,
    load_course_profiles,
    resolve_keywords,
    save_course_profiles,
)


CONFIG = {"api_base": "https://example.invalid/v1", "api_key": "test", "model": "strong-model"}


def summary_json(**overrides):
    data = {
        "main_content": "本节课讲解了欧姆定律和电容。",
        "important_items": [{"type": "作业", "description": "第5章习题", "deadline": "下周四"}],
        "attendance_quiz_events": [
            {"keyword": "小测", "matched": True, "time": "14:23:15", "context": "大家把书收起来"}
        ],
        "keyword_effectiveness": "老师常用“大家把书收起来”代替小测。",
        "keyword_adjustments": {
            "add": ["大家把书收起来"],
            "remove": [],
            "style_notes_update": "把“点名”说成“查人数”",
        },
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def completion(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content="思考"),
                finish_reason=finish_reason,
            )
        ]
    )


@contextmanager
def mock_completion(content, finish_reason="stop"):
    sdk = ModuleType("openai")
    sdk.OpenAI = Mock()
    create = sdk.OpenAI.return_value.chat.completions.create
    create.return_value = completion(content, finish_reason)
    with patch.dict("sys.modules", {"openai": sdk}):
        yield create


class ResolveKeywordsTests(unittest.TestCase):
    def test_none_inputs_return_defaults(self):
        self.assertEqual(resolve_keywords(None, None), DEFAULT_KEYWORDS)

    def test_union_dedupes_and_preserves_order(self):
        profile = {"keywords": ["查人数", "小测"]}
        result = resolve_keywords(["雷达", "小测"], profile)
        self.assertEqual(result, DEFAULT_KEYWORDS + ["查人数"])

    def test_cli_words_are_appended(self):
        result = resolve_keywords(["新词A"], None)
        self.assertEqual(result, DEFAULT_KEYWORDS + ["新词A"])


class ProfileTests(unittest.TestCase):
    def test_load_missing_and_corrupt_files(self):
        self.assertEqual(load_course_profiles("/nonexistent/x.json"), {})
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_course_profiles(str(bad)), {})

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "courses.json"
            save_course_profiles(str(path), {"87120": {"keywords": ["查人数"]}})
            self.assertEqual(
                load_course_profiles(str(path)),
                {"87120": {"keywords": ["查人数"]}},
            )

    def test_adjustments_only_remove_profile_keywords(self):
        profiles = {"87120": {"keywords": ["自定义词", "另一词"]}}
        profile, changed = apply_keyword_adjustments(
            "87120",
            profiles["87120"],
            {"add": ["新词"], "remove": ["自定义词", "点到"], "style_notes_update": ""},
            profiles,
        )
        self.assertTrue(changed)
        # "点到" is a built-in default, not in the profile, so it stays untouched.
        self.assertEqual(profile["keywords"], ["另一词", "新词"])
        self.assertIn("history", profile)
        self.assertEqual(profile["history"][-1]["add"], ["新词"])
        self.assertEqual(profile["history"][-1]["remove"], ["自定义词"])

    def test_adjustments_create_profile_when_missing(self):
        profiles = {}
        profile, changed = apply_keyword_adjustments(
            "87120", None, {"add": ["新词"], "remove": [], "style_notes_update": "风格"}, profiles
        )
        self.assertTrue(changed)
        self.assertIn("87120", profiles)
        self.assertEqual(profile["keywords"], ["新词"])
        self.assertEqual(profile["style_notes"], "风格")

    def test_empty_adjustments_do_not_change(self):
        profiles = {"87120": {"keywords": ["自定义词"]}}
        profile, changed = apply_keyword_adjustments(
            "87120", profiles["87120"], {"add": [], "remove": [], "style_notes_update": ""}, profiles
        )
        self.assertFalse(changed)
        self.assertNotIn("history", profile)


class SummaryParsingTests(unittest.TestCase):
    def test_valid_summary(self):
        s = _parse_summary_answer(summary_json())
        self.assertEqual(s.main_content, "本节课讲解了欧姆定律和电容。")
        self.assertEqual(len(s.important_items), 1)
        self.assertEqual(s.keyword_adjustments["add"], ["大家把书收起来"])

    def test_fenced_json_is_accepted(self):
        s = _parse_summary_answer("```json\n" + summary_json() + "\n```")
        self.assertEqual(s.main_content, "本节课讲解了欧姆定律和电容。")

    def test_invalid_summaries_are_rejected(self):
        cases = [
            "", "[]", "null", "{}",
            summary_json(main_content=""),
            summary_json(main_content=123),
            summary_json(important_items="bad"),
            summary_json(important_items=[{"no_description": True}]),
            summary_json(attendance_quiz_events="bad"),
            summary_json(keyword_adjustments=[]),
        ]
        for content in cases:
            with self.subTest(content=content):
                with self.assertRaises((ValueError, TypeError)):
                    _parse_summary_answer(content)


class GenerateSummaryTests(unittest.TestCase):
    def test_generation_returns_structured_summary(self):
        with mock_completion(summary_json()) as create:
            result = generate_course_summary(
                "转录文本", "87120", "大学物理", ["小测"], "", CONFIG,
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.main_content, "本节课讲解了欧姆定律和电容。")
        self.assertEqual(set(create.call_args.kwargs), {
            "model", "messages", "max_tokens", "temperature", "timeout",
        })

    def test_api_error_returns_none(self):
        with mock_completion(None) as create:
            create.side_effect = TimeoutError("test timeout")
            with self.assertLogs("src.summarizer", level="ERROR"):
                self.assertIsNone(
                    generate_course_summary("文本", "87120", "课", ["小测"], "", CONFIG)
                )

    def test_invalid_json_returns_none(self):
        with mock_completion("not json"), self.assertLogs("src.summarizer", level="ERROR"):
            self.assertIsNone(
                generate_course_summary("文本", "87120", "课", ["小测"], "", CONFIG)
            )

    def test_missing_config_returns_none(self):
        self.assertIsNone(
            generate_course_summary("文本", "87120", "课", ["小测"], "", None)
        )


class FormattingTests(unittest.TestCase):
    def test_markdown_contains_sections(self):
        s = _parse_summary_answer(summary_json())
        md = build_summary_markdown("87120", "大学物理", "2026-10-08", s, "风格记录")
        for heading in ("## 主要内容", "## 重要事项", "## 考勤与小测统计",
                        "## 关键词有效性分析", "## 关键词调整"):
            self.assertIn(heading, md)
        self.assertIn("风格记录", md)

    def test_digest_is_compact(self):
        s = _parse_summary_answer(summary_json())
        digest = build_summary_digest("87120", "大学物理", s, "logs/87120.summary.md")
        self.assertIn("[智云课堂总结]", digest)
        self.assertIn("大家把书收起来", digest)
        self.assertLessEqual(len(digest), 2800)


if __name__ == "__main__":
    unittest.main()
