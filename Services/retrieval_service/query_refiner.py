
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

    def _init_history(self):
        """إنشاء ملف سجل البحث إذا لم يكن موجوداً"""
        if not self.history_path.exists():
            with open(self.history_path, 'w') as f:
                json.dump([], f)

    def correct_spelling(self, query: str) -> str:
        """تصحيح الأخطاء الإملائية مع حماية الكلمات الصحيحة تماماً"""
        words = query.split()
        corrected_words = []
        for word in words:
            # إذا كانت الكلمة مكونة من حرفين أو ثلاثة وضمن الضمائر الشائعة لا تلمسها
            if word.lower() in ['us', 'for', 'the', 'is', 'on', 'in', 'it']:
                corrected_words.append(word)
                continue
                
            # فحص إذا كانت الكلمة معروفة للقاموس أصلاً
            if word.lower() in self.spell:
                corrected_words.append(word)
            else:
                cor = self.spell.correction(word)
                corrected_words.append(cor if cor else word)
        
        return " ".join(corrected_words)

    def expand_with_synonyms(self, raw_query: str) -> list:
        """توسيع الاستعلام بالمرادفات باستخدام الكلمات الأصلية لمنع التشوه اللغوي"""
        # تقسيم النص الأصلي إلى كلمات وتجنب الضمائر والكلمات الشائعة فوراً
        words = [w.lower() for w in raw_query.split()]
        ignored_words = ['us', 'u', 'it', 'me', 'is', 'are', 'am', 'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for']
        
        expanded_words = set()
        
        for word in words:
            if word in ignored_words or len(word) <= 2:
                continue
                
            expanded_words.add(word) # إضافة الكلمة الأصلية
            
            for syn in wordnet.synsets(word):
                # نركز فقط على الأسماء والصفات والأفعال الأكاديمية (نبتعد عن العامية)
                if syn.lexname() in ['noun.communication', 'noun.cognition', 'adj.all', 'noun.state', 'noun.phenomenon']:
                    for lemma in syn.lemmas():
                        word_name = lemma.name().lower()
                        # شروط صارمة: كلمة واحدة، بدون رموز، وحروفها نظيفة
                        if "_" not in word_name and "-" not in word_name and len(word_name) > 2:
                            expanded_words.add(word_name)
                            
        return list(expanded_words)

    def save_to_history(self, query: str):
        """حفظ الاستعلام في سجل البحث لاقتراحه لاحقاً"""
        try:
            with open(self.history_path, 'r') as f:
                history = json.load(f)
            if query not in history:
                history.append(query)
                with open(self.history_path, 'w') as f:
                    json.dump(history[-20:], f) # حفظ آخر 20 بحث فقط
        except:
            pass

    def get_suggestions(self, current_input: str) -> list:
        """اقتراح استعلامات بناءً على سجل البحث (Query Suggestion)"""
        if not current_input: return []
        with open(self.history_path, 'r') as f:
            history = json.load(f)
        return [q for q in history if q.startswith(current_input.lower())][:5]