from sievio.core.language_id import LanguageConfig
from sievio.core.records import build_record


def test_default_language_display_names():
    rec_py = build_record(text="", rel_path="notebook.ipynb")
    assert rec_py["meta"]["lang"] == "Python"

    rec_ts = build_record(text="", rel_path="component.tsx")
    assert rec_ts["meta"]["lang"] == "TypeScript"


def test_custom_language_display_names():
    custom_cfg = LanguageConfig()
    custom_cfg.display_names["python"] = "Py"

    rec = build_record(text="", rel_path="script.py", langcfg=custom_cfg)
    assert rec["meta"]["lang"] == "Py"


def test_custom_extension_mapping():
    custom_cfg = LanguageConfig()
    custom_cfg.ext_lang[".custom"] = "python"

    rec = build_record(text="", rel_path="test.custom", langcfg=custom_cfg)
    assert rec["meta"]["lang"] == "Python"
