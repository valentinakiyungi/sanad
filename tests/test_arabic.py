from sanad.arabic import detect_language, normalize, stem_ar, tokenize


def test_removes_diacritics_and_tatweel():
    assert normalize("العَمَـــل") == "العمل"


def test_unifies_alef_yaa_taa_marbuta():
    assert normalize("أجر إجازة آخر") == "اجر اجازه اخر"
    assert normalize("القصوى") == "القصوي"


def test_arabic_indic_digits():
    assert normalize("المادة (٣) و ١٤ يوماً") == "الماده (3) و 14 يوما"


def test_light_stemmer_strips_prefixes_but_keeps_short_words():
    assert stem_ar("والعامل") == "عامل"
    assert stem_ar("للمستهلك") == "مستهلك"
    assert stem_ar("ال") == "ال"  # too short to strip


def test_query_and_document_forms_meet():
    # "لفترة التجربة" in a question must match "فترة التجربة" in the article.
    assert set(tokenize("لفترة التجربة")) == set(tokenize("فترةُ التَّجربة"))


def test_drops_stopwords_both_languages():
    assert tokenize("ما هي the overtime") == ["overtime"]


def test_detect_language():
    assert detect_language("ما هي مدة فترة التجربة؟") == "ar"
    assert detect_language("What is the probation period?") == "en"
    assert detect_language("") == "en"
