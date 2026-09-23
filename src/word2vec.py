"""
word2vec.py — Word2Vec (skip-gram + negative sampling) **tự cài đặt** bằng NumPy.

Đây là kỹ thuật của **Lab 05**, cài lại từ đầu theo cùng tinh thần với
vectorizer.py (Lab 04): không gọi `gensim`, mọi công thức viết tay, và có kiểm
thử chứng minh gradient đúng (tests/test_word2vec.py).

Module này **KHÔNG được chatbot dùng** — nó phục vụ thí nghiệm so sánh ở
docs/10 (tools/exp_word2vec.py). Kết luận thí nghiệm quyết định có đưa nó vào
bot hay không, không phải ngược lại.

## Ý tưởng (Lab 05, Phần A)

TF-IDF coi "máy_bay" và "phi_cơ" là hai chiều vocabulary tách biệt -> cosine 0.
Word2Vec hỏi câu khác: *những từ nào hay xuất hiện trong ngữ cảnh giống nhau?*
Mỗi từ được học một vector dày (dense) d chiều, sao cho từ ở trung tâm dự đoán
được các từ trong cửa sổ ngữ cảnh quanh nó.

## Công thức (skip-gram + negative sampling, Mikolov et al. 2013)

Mỗi từ có hai vector: v_w (khi làm TÂM, ma trận W) và u_w (khi làm NGỮ CẢNH,
ma trận C). Với một cặp (tâm c, ngữ cảnh o) và k từ nhiễu n_1..n_k rút từ phân
phối unigram^0.75, hàm mất mát là:

    L = -log σ(u_o · v_c)  -  Σ_i log σ(-u_{n_i} · v_c)

Kéo cặp thật lại gần nhau, đẩy cặp nhiễu ra xa. Đạo hàm (đặt s = σ(u_o·v_c),
t_i = σ(u_{n_i}·v_c)):

    ∂L/∂v_c     = (s - 1)·u_o + Σ_i t_i·u_{n_i}
    ∂L/∂u_o     = (s - 1)·v_c
    ∂L/∂u_{n_i} = t_i·v_c

Hai kỹ thuật phụ của bản gốc cũng được giữ:
  - cửa sổ ĐỘNG: mỗi từ tâm rút kích thước cửa sổ b ∈ [1, window], nên từ gần
    tâm được lấy mẫu nhiều hơn từ xa;
  - SUBSAMPLING từ quá phổ biến: giữ từ w với xác suất (√(z/s) + 1)·s/z, z là
    tần suất tương đối, s = `sample`.

Cập nhật theo lô nhỏ (mini-batch): gradient của cả lô được CỘNG DỒN vào từng
dòng — cùng một từ xuất hiện nhiều lần trong lô thì cộng đủ số lần, không bị
ghi đè (xem _scatter_add).
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Cắt biên để exp không tràn số; σ(±15) đã sát 0/1 tới ~3e-7.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def _scatter_add(M: np.ndarray, idx: np.ndarray, G: np.ndarray) -> None:
    """M[idx] += G, CỘNG DỒN khi idx có phần tử trùng — cùng ngữ nghĩa np.add.at.

    np.add.at đúng nhưng rất chậm (vòng lặp không đệm bên trong). Ở đây sắp xếp
    chỉ số, cộng từng đoạn liên tiếp bằng np.add.reduceat, rồi ghi một lần cho
    mỗi dòng duy nhất: nhanh hơn cỡ một bậc, kết quả như nhau (chỉ khác thứ tự
    cộng số thực). Kiểm chứng trong tests/test_word2vec.py.
    """
    if idx.size == 0:
        return
    order = np.argsort(idx, kind="stable")
    idx_s = idx[order]
    starts = np.flatnonzero(np.r_[True, idx_s[1:] != idx_s[:-1]])
    M[idx_s[starts]] += np.add.reduceat(G[order], starts, axis=0)


def sgns_loss_and_grads(v_c: np.ndarray, u_o: np.ndarray, u_n: np.ndarray):
    """Mất mát và gradient của skip-gram negative sampling cho MỘT LÔ.

    v_c: (B, d) vector tâm; u_o: (B, d) vector ngữ cảnh thật; u_n: (B, k, d)
    vector nhiễu. Trả (loss trung bình, dv_c, du_o, du_n) — gradient chưa chia
    cho B, tức là gradient của TỔNG mất mát, đúng thứ SGD cần.

    Tách thành hàm thuần để kiểm thử so với đạo hàm số (tests/test_word2vec.py).
    """
    s = _sigmoid(np.einsum("bd,bd->b", v_c, u_o))            # (B,)
    t = _sigmoid(np.einsum("bkd,bd->bk", u_n, v_c))          # (B, k)
    loss = -np.log(s + 1e-12) - np.log(1.0 - t + 1e-12).sum(axis=1)
    g = (s - 1.0)[:, None]
    dv_c = g * u_o + np.einsum("bk,bkd->bd", t, u_n)
    du_o = g * v_c
    du_n = t[:, :, None] * v_c[:, None, :]
    return float(loss.mean()), dv_c, du_o, du_n


class Word2Vec:
    """Word2Vec skip-gram + negative sampling. API gần với gensim để đối chiếu Lab 05."""

    def __init__(
        self,
        vector_size: int = 50,
        window: int = 5,
        min_count: int = 1,
        negative: int = 5,
        epochs: int = 30,
        lr: float = 0.025,
        min_lr: float = 1e-4,
        sample: float = 1e-3,
        batch_size: int = 512,
        seed: int = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.lr = lr
        self.min_lr = min_lr
        self.sample = sample
        self.batch_size = batch_size
        self.seed = seed

        self.vocab: dict[str, int] = {}
        self.index_to_word: list[str] = []
        self.counts: np.ndarray | None = None
        self.W: np.ndarray | None = None      # vector TÂM — đây là "word vector"
        self.C: np.ndarray | None = None      # vector NGỮ CẢNH
        self.loss_history: list[float] = []
        self._normed: np.ndarray | None = None

    # -- dựng từ vựng --------------------------------------------------------
    def _build_vocab(self, sentences: list[list[str]]) -> None:
        counter = Counter(tok for sent in sentences for tok in sent)
        kept = sorted((w for w, c in counter.items() if c >= self.min_count),
                      key=lambda w: (-counter[w], w))    # ổn định giữa các lần chạy
        self.index_to_word = kept
        self.vocab = {w: i for i, w in enumerate(kept)}
        self.counts = np.array([counter[w] for w in kept], dtype=np.float64)

    def _pairs_for_epoch(self, encoded: list[np.ndarray], rng) -> np.ndarray:
        """Sinh toàn bộ cặp (tâm, ngữ cảnh) của một epoch: subsampling + cửa sổ động."""
        total = self.counts.sum()
        z = self.counts / total
        keep_p = np.minimum(1.0, (np.sqrt(z / self.sample) + 1.0) * self.sample / z) \
            if self.sample else np.ones_like(z)

        chunks = []
        for arr in encoded:
            if self.sample:
                arr = arr[rng.random(arr.size) < keep_p[arr]]
            L = arr.size
            if L < 2:
                continue
            b = rng.integers(1, self.window + 1, size=L)       # cửa sổ động cho mỗi tâm
            for off in range(1, min(self.window, L - 1) + 1):
                left, right = arr[:-off], arr[off:]
                fwd = b[:-off] >= off        # tâm ở trái, ngữ cảnh ở phải
                bwd = b[off:] >= off         # tâm ở phải, ngữ cảnh ở trái
                chunks.append(np.stack([left[fwd], right[fwd]], axis=1))
                chunks.append(np.stack([right[bwd], left[bwd]], axis=1))
        if not chunks:
            return np.empty((0, 2), dtype=np.int64)
        pairs = np.concatenate(chunks)
        rng.shuffle(pairs)
        return pairs

    # -- huấn luyện ----------------------------------------------------------
    def fit(self, sentences: list[list[str]], verbose: bool = False) -> "Word2Vec":
        rng = np.random.default_rng(self.seed)
        self._build_vocab(sentences)
        V, d = len(self.index_to_word), self.vector_size
        if V == 0:
            raise ValueError("Từ vựng rỗng — kiểm tra min_count hoặc dữ liệu vào.")

        # Khởi tạo như bản C gốc: W ngẫu nhiên nhỏ, C bằng 0. Dùng float32 như
        # gensim và bản C: độ chính xác thừa đủ cho SGD, còn băng thông bộ nhớ
        # giảm một nửa — huấn luyện ở đây bị giới hạn bởi băng thông.
        self.W = ((rng.random((V, d)) - 0.5) / d).astype(np.float32)
        self.C = np.zeros((V, d), dtype=np.float32)

        encoded = [np.array([self.vocab[t] for t in s if t in self.vocab], dtype=np.int64)
                   for s in sentences]
        # Phân phối nhiễu unigram^0.75: nâng nhẹ từ hiếm so với tần suất thật.
        noise = self.counts ** 0.75
        noise /= noise.sum()
        noise_cdf = np.cumsum(noise)

        # Sinh cặp theo TỪNG epoch (không giữ cả 30 epoch trong RAM). Tổng số
        # bước cho lịch giảm tốc độ học ước lượng từ epoch đầu — các epoch chỉ
        # khác nhau do rút ngẫu nhiên nên số cặp xấp xỉ nhau.
        pairs = self._pairs_for_epoch(encoded, rng)
        total_steps = max(1, pairs.shape[0] * self.epochs)
        done = 0
        self.loss_history = []
        for ep in range(1, self.epochs + 1):
            if ep > 1:
                pairs = self._pairs_for_epoch(encoded, rng)
            losses = []
            for start in range(0, pairs.shape[0], self.batch_size):
                batch = pairs[start:start + self.batch_size]
                c, o = batch[:, 0], batch[:, 1]
                neg = np.searchsorted(noise_cdf, rng.random((c.size, self.negative)))
                neg = np.minimum(neg, V - 1)
                # Tốc độ học giảm tuyến tính theo tiến độ, như bản gốc.
                lr = max(self.min_lr, self.lr * (1.0 - done / total_steps))
                loss, dv, du_o, du_n = sgns_loss_and_grads(self.W[c], self.C[o], self.C[neg])
                _scatter_add(self.W, c, -lr * dv)
                _scatter_add(self.C, o, -lr * du_o)
                _scatter_add(self.C, neg.ravel(), -lr * du_n.reshape(-1, d))
                losses.append(loss)
                done += c.size
            self.loss_history.append(float(np.mean(losses)) if losses else float("nan"))
            if verbose:
                print(f"  epoch {ep:>2}/{self.epochs}: {pairs.shape[0]:>8,} cặp, "
                      f"loss {self.loss_history[-1]:.4f}")
        self._normed = None
        return self

    # -- tra cứu (giống gensim.wv) -------------------------------------------
    def __contains__(self, word: str) -> bool:
        return word in self.vocab

    def __len__(self) -> int:
        return len(self.index_to_word)

    def vector(self, word: str) -> np.ndarray:
        return self.W[self.vocab[word]]

    def _unit(self) -> np.ndarray:
        if self._normed is None:
            norms = np.linalg.norm(self.W, axis=1, keepdims=True)
            self._normed = self.W / np.where(norms == 0, 1.0, norms)
        return self._normed

    def similarity(self, a: str, b: str) -> float:
        U = self._unit()
        return float(U[self.vocab[a]] @ U[self.vocab[b]])

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        U = self._unit()
        sims = U @ U[self.vocab[word]]
        sims[self.vocab[word]] = -np.inf
        best = np.argsort(-sims)[:topn]
        return [(self.index_to_word[i], float(sims[i])) for i in best]

    def document_vector(self, tokens: list[str], weights: dict[str, float] | None = None
                        ) -> np.ndarray:
        """Vector tài liệu = TRUNG BÌNH vector các từ có trong từ vựng (Lab 05, Phần 4).

        `weights` (ví dụ idf) cho phép trung bình có trọng số — một biến thể phổ
        biến, ghép Lab 04 (IDF) với Lab 05 (vector dày). Câu không có từ nào
        trong từ vựng trả vector 0 (cosine với mọi bài = 0).
        """
        rows, ws = [], []
        for t in tokens:
            idx = self.vocab.get(t)
            if idx is not None:
                rows.append(idx)
                ws.append(weights.get(t, 1.0) if weights else 1.0)
        if not rows:
            return np.zeros(self.vector_size)
        w = np.asarray(ws)
        return (self.W[rows] * w[:, None]).sum(axis=0) / w.sum()

    def coverage(self, tokens: list[str]) -> float:
        """Tỷ lệ token có vector (Lab 05, Phần 5)."""
        return sum(t in self.vocab for t in tokens) / len(tokens) if tokens else 0.0
