"""
Requirement 5: Query Refinement Service.
Handles Spelling Correction, Query Expansion (Synonyms), and Search History.
"""

import json
from pathlib import Path
from spellchecker import SpellChecker
from nltk.corpus import wordnet


class QueryRefiner:
    def __init__(self, history_path: Path):
        self.spell = SpellChecker()
        self.history_path = history_path
        self._init_history()

        # قائمة موحدة وشاملة لحماية الكلمات الشائعة من التصحيح الإملائي العشوائي ومن جلب المرادفات المشوهة
        self.protected_stopwords = {
            "us",
            "u",
            "it",
            "me",
            "my",
            "we",
            "you",
            "they",
            "he",
            "she",
            "him",
            "her",
            "is",
            "are",
            "am",
            "was",
            "were",
            "be",
            "been",
            "being",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "by",
            "for",
            "to",
            "of",
            "and",
            "or",
            "not",
            "but",
            "making",
            "make",
            "do",
            "does",
            "did",
            "with",
            "about",
            "from",
            "as",
        }

    def _init_history(self):
        """إنشاء ملف سجل البحث إذا لم يكن موجوداً"""
        if not self.history_path.exists():
            with open(self.history_path, "w") as f:
                json.dump([], f)

    def correct_spelling(self, query: str) -> str:
        """تصحيح الأخطاء الإملائية مع حماية الكلمات الصحيحة والضمائر الشائعة"""
        words = query.split()
        corrected_words = []

        for word in words:
            # تنظيف الكلمة من علامات الاستفهام أو النقاط لتفادي تشتيت المصحح
            clean_word = word.lower().strip("?!.,:;")

            # إذا كانت الكلمة محمية ضمن القائمة الموحدة، نتركها فوراً بدون تعديل
            if clean_word in self.protected_stopwords:
                corrected_words.append(word)
                continue

            # إذا كانت الكلمة معروفة للقاموس الأصلي نتركها
            if clean_word in self.spell:
                corrected_words.append(word)
            else:
                cor = self.spell.correction(clean_word)
                corrected_words.append(cor if cor else word)

        return " ".join(corrected_words)

    def expand_with_synonyms(self, raw_query: str) -> list:
        """توسيع الاستعلام بالمرادفات النظيفة والأكاديمية باستخدام الكلمات الأصلية الفصيحة"""
        words = raw_query.split()
        expanded_words = set()

        # Moved outside the loop and converted to a set for better performance
        allowed_lexnames = {
            "noun.communication",
            "noun.cognition",
            "adj.all",
            "noun.state",
            "noun.phenomenon",
            "noun.act",
            "verb.cognition",
            "adv.all",
        }

        for word in words:
            clean_word = word.lower().strip("?!.,:;")

            # استبعاد الكلمات الوظيفية والقصيرة جداً فوراً بناءً على القائمة الموحدة
            if clean_word in self.protected_stopwords or len(clean_word) <= 2:
                continue

            # إضافة الكلمة الأصلية النظيفة أولاً
            expanded_words.add(clean_word)

            # جلب المرادفات الأكاديمية فقط من WordNet
            for syn in wordnet.synsets(clean_word):
                # FIX: Added a type-checker safety check (syn is not None and hasattr)
                if syn is not None and hasattr(syn, "lexname"):
                    if syn.lexname() in allowed_lexnames:
                        for lemma in syn.lemmas():
                            word_name = lemma.name().lower()
                            # تصفية الكلمات المركبة التي تحتوي على "_" أو "-" للتأكد من ملاءمتها للـ Index
                            if (
                                "_" not in word_name
                                and "-" not in word_name
                                and len(word_name) > 2
                            ):
                                expanded_words.add(word_name)

        return list(expanded_words)

    def save_to_history(self, query: str):
        """حفظ الاستعلام في سجل البحث لاقتراحه لاحقاً"""
        try:
            with open(self.history_path, "r") as f:
                history = json.load(f)
            if query not in history:
                history.append(query)
                with open(self.history_path, "w") as f:
                    json.dump(history[-20:], f)  # حفظ آخر 20 بحث فقط
        except:
            pass

    def get_suggestions(self, current_input: str) -> list:
        """اقتراح استعلامات بناءً على سجل البحث (Query Suggestion)"""
        if not current_input:
            return []
        try:
            with open(self.history_path, "r") as f:
                history = json.load(f)
            return [q for q in history if q.startswith(current_input.lower())][:5]
        except:
            return []
