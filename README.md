# 🧠 Simple Autocomplete Model with TF-IDF + FastAPI

> A lightweight, character-based autocomplete engine using TF-IDF and cosine similarity — built with scikit-learn and FastAPI.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)

---

## ✨ Features

- ✅ Suggests words based on prefix input (e.g., “aut” → “auto”, “auth”, “autism”)
- ✅ Uses **character n-grams** (2–4 chars) for fuzzy matching
- ✅ Powered by **TF-IDF + Cosine Similarity**
- ✅ Served via **FastAPI** — ready for web integration
- ✅ Works with any word list (loaded from `words.json`)

---

## 🚀 How It Works

1. **Training**:
   - Loads a list of words from `words.json`
   - Converts each word into character n-gram TF-IDF vectors (e.g., “auto” → “ a”, “au”, “aut”, “uto”, etc.)

2. **Querying**:
   - User types a prefix (e.g., “aut”)
   - Model finds all words that start with that prefix
   - Ranks them by cosine similarity between prefix and word vectors
   - Returns top-k suggestions

> 💡 Why character n-grams?  
> They capture partial matches and typos better than full words — great for autocomplete!

---

## 📦 Installation

```bash
# Clone repo (if not already)
git clone https://github.com/Akshat1931/word_autocompleter_ml.git

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📄 Requirements (`requirements.txt`)

```txt
fastapi
uvicorn
scikit-learn
numpy
```

---

## 🗃️ Dataset

Your model expects a `words.json` file in the root directory, structured like:

```json
{
  "apple": 1,
  "application": 1,
  "apply": 1,
  "banana": 1,
  ...
}
```

> 💡 You can generate this from any word list — even `/usr/share/dict/words` on Linux/macOS!

---

## ▶️ Run the Server

```bash
uvicorn main:app --reload
```

Visit: http://localhost:8000/docs to see the interactive API docs!

---

## 🧪 Try It Out

### GET Request:
```
http://localhost:8000/suggest?prefix=aut
```

### Response:
```json
{
  "prefix": "aut",
  "suggestions": [
    "auto",
    "auth",
    "autism",
    "autos",
    "autist"
  ]
}
```

---

## 🧑‍💻 Code Structure

```
.
├── main.py          # FastAPI app + model loading
├── model.py         # AutocompleteModel class
├── words.json       # Word dataset
└── README.md
```

---

## 🛠️ Future Improvements

- Add caching for frequent prefixes
- Support typo tolerance (e.g., “aot” → “auto”)
- Deploy with Docker or on Vercel/Render
- Add frontend demo (HTML + JS)

---

## 📎 License

MIT — Feel free to use, modify, and learn from it!
