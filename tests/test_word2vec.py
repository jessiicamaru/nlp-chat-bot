"""
test_word2vec.py — Chứng minh Word2Vec tự cài đặt (src/word2vec.py) là ĐÚNG.

Không có gensim để đối chiếu (và cố ý không thêm phụ thuộc), nên dùng hai cách
kiểm chứng chuẩn cho một mô hình học bằng gradient:

  1. KIỂM TRA GRADIENT BẰNG SAI PHÂN: so gradient giải tích với đạo hàm số
     (L(x + h) - L(x - h)) / 2h trên từng tọa độ. Nếu công thức đạo hàm sai
     một dấu hay thiếu một hạng, sai số lệch hẳn khỏi ~1e-9.
  2. KHO ĐỒ CHƠI CÓ ĐÁP ÁN BIẾT TRƯỚC: hai nhóm từ không bao giờ xuất hiện
     chung ngữ cảnh (con vật / thiết bị) — giống corpus mẫu của Lab 05.
     Huấn luyện xong, từ phải gần từ CÙNG nhóm hơn từ KHÁC nhóm.

Chạy:  .venv/Scripts/python.exe tests/test_word2vec.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from word2vec import Word2Vec, _scatter_add, sgns_loss_and_grads  # noqa: E402

PASS, FAIL = 0, 0


def report(ok: bool, msg: str) -> None:
    global PASS, FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    PASS += ok
    FAIL += not ok


def total_loss(v_c, u_o, u_n):
    """Tổng (không chia B) — khớp với gradient mà sgns_loss_and_grads trả về."""
    mean, *_ = sgns_loss_and_grads(v_c, u_o, u_n)
    return mean * v_c.shape[0]


def test_gradient_check():
    print("\n--- gradient giải tích vs sai phân trung tâm ---")
    rng = np.random.default_rng(0)
    B, k, d = 4, 3, 6
    v_c = rng.normal(size=(B, d))
    u_o = rng.normal(size=(B, d))
    u_n = rng.normal(size=(B, k, d))
    _, dv, du_o, du_n = sgns_loss_and_grads(v_c, u_o, u_n)

    h = 1e-6
    for name, arr, grad in (("dL/dv_c", v_c, dv), ("dL/du_o", u_o, du_o), ("dL/du_n", u_n, du_n)):
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + h
            plus = total_loss(v_c, u_o, u_n)
            arr[idx] = old - h
            minus = total_loss(v_c, u_o, u_n)
            arr[idx] = old
            num[idx] = (plus - minus) / (2 * h)
        err = float(np.max(np.abs(num - grad)))
        report(err < 1e-6, f"{name:<8} sai số lớn nhất = {err:.2e}")


def toy_corpus(repeat=60, seed=1):
    """Hai 'thế giới' tách biệt: con vật và thiết bị, mỗi bên có ngữ cảnh riêng."""
    rng = np.random.default_rng(seed)
    animals, animal_ctx = ["mèo", "chó", "gà", "vịt"], ["ăn", "ngủ", "chạy", "kêu", "lông"]
    devices, device_ctx = ["iphone", "laptop", "tablet", "tivi"], ["pin", "sạc", "màn_hình", "chip", "cài_đặt"]
    sents = []
    for _ in range(repeat):
        for subj, ctx in ((animals, animal_ctx), (devices, device_ctx)):
            s = [subj[rng.integers(len(subj))]]
            for _ in range(4):
                s += [ctx[rng.integers(len(ctx))], subj[rng.integers(len(subj))]]
            sents.append(s)
    return sents, animals, devices


def test_toy_clusters():
    print("\n--- kho đồ chơi: từ cùng nhóm phải gần nhau hơn ---")
    sents, animals, devices = toy_corpus()
    m = Word2Vec(vector_size=16, window=2, min_count=1, epochs=40, sample=0, seed=42).fit(sents)

    report(m.loss_history[-1] < m.loss_history[0],
           f"loss giảm: {m.loss_history[0]:.3f} -> {m.loss_history[-1]:.3f}")

    same = np.mean([m.similarity(a, b) for a in animals for b in animals if a < b]
                   + [m.similarity(a, b) for a in devices for b in devices if a < b])
    cross = np.mean([m.similarity(a, b) for a in animals for b in devices])
    report(same > cross + 0.3, f"cosine cùng nhóm {same:.3f} > khác nhóm {cross:.3f}")

    top = [w for w, _ in m.most_similar("mèo", topn=3)]
    report(all(w in animals or w in ("ăn", "ngủ", "chạy", "kêu", "lông") for w in top),
           f"most_similar('mèo') toàn từ thế giới con vật: {top}")

    doc_a = m.document_vector(["mèo", "ăn", "ngủ"])
    doc_b = m.document_vector(["chó", "chạy", "kêu"])
    doc_c = m.document_vector(["laptop", "pin", "sạc"])
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    report(cos(doc_a, doc_b) > cos(doc_a, doc_c),
           f"vector tài liệu: cos(mèo…, chó…) {cos(doc_a, doc_b):.3f} > "
           f"cos(mèo…, laptop…) {cos(doc_a, doc_c):.3f}")


def test_scatter_add_matches_add_at():
    print("\n--- cộng dồn nhanh (reduceat) vs np.add.at ---")
    rng = np.random.default_rng(3)
    idx = rng.integers(0, 20, size=500)          # cố ý nhiều chỉ số TRÙNG
    G = rng.normal(size=(500, 7))
    ref = np.zeros((20, 7)); np.add.at(ref, idx, G)
    ours = np.zeros((20, 7)); _scatter_add(ours, idx, G)
    err = float(np.max(np.abs(ref - ours)))
    report(err < 1e-12, f"khớp np.add.at khi chỉ số trùng lặp, sai số {err:.1e}")


def test_oov_and_determinism():
    print("\n--- OOV và tính tái lập ---")
    sents, _, _ = toy_corpus(repeat=10)
    m1 = Word2Vec(vector_size=8, epochs=3, seed=7).fit(sents)
    m2 = Word2Vec(vector_size=8, epochs=3, seed=7).fit(sents)
    report(np.array_equal(m1.W, m2.W), "cùng seed -> cùng vector (tái lập được)")
    v = m1.document_vector(["từ_không_có", "cũng_không_có"])
    report(np.all(v == 0), "câu toàn từ OOV -> vector 0, không lỗi")
    report(m1.coverage(["mèo", "từ_lạ"]) == 0.5, "coverage đếm đúng tỷ lệ từ có vector")
    m3 = Word2Vec(min_count=1000, epochs=1)
    try:
        m3.fit(sents)
        report(False, "min_count quá cao phải báo lỗi rõ ràng")
    except ValueError:
        report(True, "min_count quá cao -> ValueError rõ ràng, không chạy âm thầm")


if __name__ == "__main__":
    print("=" * 74)
    print("KIỂM CHỨNG WORD2VEC TỰ CÀI ĐẶT (skip-gram + negative sampling)")
    print("=" * 74)
    test_gradient_check()
    test_toy_clusters()
    test_scatter_add_matches_add_at()
    test_oov_and_determinism()
    print("\n" + "=" * 74)
    print(f"KẾT QUẢ: {PASS} pass / {FAIL} fail")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
