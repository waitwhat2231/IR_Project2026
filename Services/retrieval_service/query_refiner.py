"""
Requirement 5: Query Refinement Service.
Handles Spelling Correction, Query Expansion (Synonyms), and Search History.
"""

import json
from pathlib import Path
from typing import Iterable

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

        # مفردات المتن (Corpus Vocabulary) -- تُملأ لاحقاً عبر load_corpus_vocabulary()
        # باستخدام مصطلحات الفهرس المقلوب الفعلية لكل Dataset تم تحميله.
        # الغرض: أي كلمة موجودة فعلياً في الوثائق المفهرسة (أسماء خاصة، مصطلحات نوعية..)
        # تُعتبر "معروفة" فلا يُعاد كتابتها خطأً إلى أقرب كلمة عامة في القاموس الإنكليزي.
        # مثال حقيقي وقع فعلاً: "Hitler" كانت تُصحَّح إلى "hitter" لأن مدقق الإملاء
        # العام لا يعرف أي أسماء خاصة، و"hitter" أقرب كلمة معروفة له بفارق حرف واحد فقط.
        self._corpus_vocab: set = set()

    def _init_history(self):
        """إنشاء ملف سجل البحث إذا لم يكن موجوداً"""
        if not self.history_path.exists():
            with open(self.history_path, "w") as f:
                json.dump([], f)

    def load_corpus_vocabulary(self, terms: Iterable[str]) -> None:
        """
        حقن مفردات المتن (Corpus Vocabulary) داخل المدقق الإملائي.

        تُستخدم كقائمة حماية إضافية: أي كلمة موجودة فعلياً في الوثائق المفهرسة
        (بما فيها الأسماء الخاصة والمصطلحات النوعية التي لا يعرفها قاموس اللغة
        الإنكليزية العام -- مثل "Hitler" أو "COVID") تُعتبر "معروفة" ولا يُعاد
        كتابتها إلى أقرب كلمة عامة بالخطأ.

        لا تُنشئ هذه الدالة أي تحميل جديد من القرص -- يُفترض أن `terms` هي
        مفاتيح فهرس مقلوب محمَّل بالفعل في الذاكرة (مثال:
        `hybrid.bm25.index.keys()` من InvertedIndexManager الذي يُحمَّل أصلاً
        من قبل HybridRetriever)، فالاستدعاء رخيص جداً (إضافة إلى set فقط).

        تستخدم update() عمداً (لا استبدال) بحيث يمكن استدعاؤها لأكثر من
        Dataset دون أن يفقد أي منها مفرداته إن أصبح للنظام مستقبلاً أكثر من
        مجموعة بيانات محمَّلة في الذاكرة في الوقت نفسه.
        """
        if terms:
            self._corpus_vocab.update(t.lower() for t in terms)

    def correct_spelling(self, query: str) -> str:
        """
        تصحيح الأخطاء الإملائية مع حماية:
          1) كلمات الوقف الشائعة (protected_stopwords).
          2) الكلمات المعروفة لقاموس اللغة العام (pyspellchecker).
          3) مفردات المتن المفهرس فعلياً (corpus vocabulary, إن وُجدت) --
             يحمي الأسماء الخاصة والمصطلحات النوعية الموجودة في الوثائق.
          4) أي كلمة تبدأ بحرف كبير في وسط الاستعلام (ليست أول كلمة) --
             مؤشر قوي على أنها اسم خاص لا يعرفه القاموس العام أصلاً، فلا
             تُسلَّم لمصحح الإملاء الذي سيستبدلها بأقرب كلمة عامة بنفس عدد
             التعديلات الحرفية (المثال الحقيقي: "Hitler" -> "hitter").
             نتجاهل أول كلمة في الاستعلام لأن بداية الجملة تُكتب بحرف كبير
             في الإنكليزية بغض النظر عن كون الكلمة اسماً خاصاً أم لا (مثال:
             "Should teachers get tenure?") -- معالجتها بشكل خاص هنا كانت
             ستُسقط تصحيح الأخطاء الإملائية الحقيقية في أول كلمة من السؤال.
        """
        words = query.split()
        corrected_words = []

        for idx, word in enumerate(words):
            # تنظيف الكلمة من علامات الاستفهام أو النقاط لتفادي تشتيت المصحح
            clean_word = word.lower().strip("?!.,:;")

            # (1) كلمة محمية ضمن القائمة الموحدة -> تترك فوراً بدون تعديل
            if clean_word in self.protected_stopwords:
                corrected_words.append(word)
                continue

            # (2) كلمة معروفة للقاموس الأصلي -> تترك بدون تعديل
            if clean_word in self.spell:
                corrected_words.append(word)
                continue

            # (3) كلمة موجودة فعلياً في مفردات المتن المفهرس -> تترك بدون تعديل
            if clean_word in self._corpus_vocab:
                corrected_words.append(word)
                continue

            # (4) كلمة تبدأ بحرف كبير ولا تقع في أول الاستعلام -> اسم خاص محتمل
            if idx > 0 and word[:1].isupper():
                corrected_words.append(word)
                continue

            # (5) غير ذلك -> نطبّق التصحيح الإملائي الفعلي
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