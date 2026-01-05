import utils


def test_split_lang_suffix_no_underscore():
    base, lang = utils.split_lang_suffix("video")
    assert base == "video"
    assert lang is None


def test_split_lang_suffix_with_lang_2_letters():
    base, lang = utils.split_lang_suffix("video_en")
    assert base == "video"
    assert lang == "en"


def test_split_lang_suffix_with_lang_hyphen():
    base, lang = utils.split_lang_suffix("video_pt-BR")
    assert base == "video"
    assert lang == "pt-br"


def test_split_lang_suffix_ignores_non_lang_suffix():
    base, lang = utils.split_lang_suffix("video_part_1")
    assert base == "video_part_1"
    assert lang is None
