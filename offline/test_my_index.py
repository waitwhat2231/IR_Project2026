import sys
import time  
from pathlib import Path

# إضافة مسار المشروع الرئيسي
sys.path.append(str(Path(__file__).parent.parent))
from Services.indexing_service.inverted_index import InvertedIndexManager

print("[*] Testing Majd's updated InvertedIndexManager...")
# استدعاء المدير الخاص بالداتاسيت
manager = InvertedIndexManager("webis-touche2020")

# طباعة الإحصائيات الشاملة التي احتسبها
print(f"\n--- Corpus Statistics ---")
print(f"[+] Total Documents (N): {manager.N:,}")
print(f"[+] Average Doc Length (avg_dl): {manager.avg_dl:.2f} tokens")
print(f"[+] Total Unique Terms: {len(manager.index):,}")

# فحص كلمة وحساب الـ IDF الرياضي لها
word = "health"
start_time = time.perf_counter()
postings = manager.get_postings(word)
end_time = time.perf_counter()
execution_time_ms = (end_time - start_time) * 1000
print(f"\n--- Testing Word: '{word}' ---")
print(f" -> Document Frequency (DF): {len(postings):,}")
print(f" -> Computed BM25 IDF: {manager.idf(word):.4f}")
print(f" -> Time Taken: {execution_time_ms:.4f} ms")