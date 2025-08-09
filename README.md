# 🌍 Multilingual Natural Language Processing (MNLP) Coursework

A comprehensive collection of seven Natural Language Processing tasks implementing techniques for multilingual text understanding, semantic similarity, and visual-linguistic integration.

## 📋 Table of Contents

- [Overview](#overview)
- [Tasks Implemented](#tasks-implemented)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Task Details](#task-details)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Reports](#reports)
- [License](#license)

## 🎯 Overview

This repository contains implementations of seven distinct NLP tasks developed as part of advanced coursework in Multilingual Natural Language Processing. Each task explores different aspects of language understanding, from token-level classification to semantic similarity and cross-modal reasoning.

### 🏆 Key Achievements

- **Multi-transformer Architecture**: Implementation of BERT, RoBERTa, DistilBERT, and ALBERT models
- **Multilingual Support**: Language detection across 20+ languages including high and low-resource languages
- **Cross-modal Integration**: Visual-linguistic understanding combining CLIP and BLIP models
- **State-of-the-art Performance**: Competitive results on benchmark datasets
- **Comprehensive Evaluation**: Detailed performance analysis with academic reports

## 🚀 Tasks Implemented

### 1. 🏷️ Named Entity Recognition (NER)
**File**: `EntityRecognition.ipynb`

Table-filling approach for named entity recognition using transformer models with BIO tagging scheme.

**Key Features**:
- Multiple transformer models (DistilBERT, BERT, RoBERTa, ALBERT)
- PyTorch Lightning framework for robust training
- BIO tagging scheme for entity boundaries
- Comprehensive evaluation metrics

**Performance**: Detailed results available in [Entity Recognition Report](reports/EntityRecognition_report.pdf)

### 2. 🔍 Word Sense Disambiguation (WSD)
**File**: `WordSenseDisambiguation.ipynb`

Advanced WSD implementation using contextual embeddings and semantic similarity.

**Key Features**:
- BERT-based contextual embeddings
- Gloss-context similarity matching
- Multi-language support
- WordNet integration

**Performance**: Comprehensive analysis in [Word Sense Disambiguation Report](reports/WordSenseDisambiguation_report.pdf)

### 3. ❓ Question Answering System
**File**: `QuestionAnswering.ipynb`

Intelligent QA system for semantic property extraction from knowledge bases.

**Key Features**:
- GloVe embeddings for semantic similarity
- RAKE keyword extraction
- Transformer-based QA pipelines
- Support for yes/no and extractive questions
- Animal and artifact domain specialization

**Datasets**:
- Animal properties (93 items, 44 questions each)
- Artifact properties (134 items, 55 questions each)

### 4. 🌐 Language Classification
**File**: `LanguageClassification.ipynb`

Dual-model approach for high-resource and low-resource language detection.

**Key Features**:
- XLM-RoBERTa base model
- 20 high-resource languages
- Custom classifier architecture
- Extensive preprocessing and tokenization

**Languages Supported**:
- High-resource: English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese, Arabic, Hindi, Turkish, Thai, Vietnamese, Polish, Bulgarian, Greek, Urdu, Swahili

### 5. 📅 Event Detection
**File**: `EventDetection.ipynb`

Sequence labeling for temporal event identification in text.

**Key Features**:
- Custom neural architecture
- Token-level classification
- JSON-based dataset processing
- Comprehensive evaluation metrics

### 6. 📊 SimLex Semantic Similarity
**File**: `SimLex.ipynb`

Word embedding evaluation using the SimLex-999 benchmark.

**Key Features**:
- Word2Vec model training
- Sense-aware embeddings
- Correlation analysis with human judgments
- Multiple embedding dimensions and techniques

### 7. 👁️ Visual Word Sense Disambiguation
**File**: `VisualWordSenseDisambiguation.ipynb`

Multi-modal approach combining visual and textual information for WSD.

**Key Features**:
- CLIP model for image-text alignment
- BLIP model for visual question answering
- Multi-lingual sentence transformers
- Cross-modal semantic understanding
- WordNet integration for sense definitions

**Models Used**:
- CLIP (Contrastive Language-Image Pre-training)
- BLIP (Bootstrapping Language-Image Pre-training)
- Sentence Transformers for multilingual support

## 🛠️ Technologies Used

### Core Frameworks
- **PyTorch** & **PyTorch Lightning**: Deep learning framework
- **Transformers (HuggingFace)**: Pre-trained language models
- **NLTK**: Natural language processing toolkit
- **spaCy**: Advanced NLP library

### Models and Architectures
- **BERT/DistilBERT/RoBERTa/ALBERT**: Transformer language models
- **XLM-RoBERTa**: Multilingual transformer
- **CLIP**: Vision-language model
- **BLIP**: Bootstrapped vision-language model
- **Word2Vec/GloVe**: Word embeddings

### Libraries and Tools
- **scikit-learn**: Machine learning utilities
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **tqdm**: Progress tracking
- **Google Colab**: Cloud computing environment

## 📁 Project Structure

```
MNLP/
├── EntityRecognition.ipynb                    # Named Entity Recognition
├── WordSenseDisambiguation.ipynb             # Word Sense Disambiguation
├── QuestionAnswering.ipynb                   # Question Answering System
├── LanguageClassification.ipynb              # Language Detection
├── EventDetection.ipynb                      # Event Detection
├── SimLex.ipynb                              # Semantic Similarity Evaluation
├── VisualWordSenseDisambiguation.ipynb       # Visual WSD
├── reports/
│   ├── EntityRecognition_report.pdf          # NER Task Report
│   └── WordSenseDisambiguation_report.pdf    # WSD Task Report
└── README.md                                 # This file
```

## 📖 Task Details

### Named Entity Recognition
- **Approach**: Table-filling with transformer encoders
- **Models**: DistilBERT, BERT, RoBERTa, ALBERT
- **Evaluation**: Token-level F1, precision, recall
- **Framework**: PyTorch Lightning with proper seeding

### Word Sense Disambiguation
- **Method**: Contextual similarity between glosses and contexts
- **Embeddings**: BERT-based contextual representations
- **Evaluation**: Accuracy on standard WSD benchmarks
- **Integration**: WordNet synset mapping

### Question Answering
- **Strategy**: Similarity-based answer extraction
- **Features**: RAKE keyword extraction, GloVe embeddings
- **Domains**: Animal and artifact property questions
- **Output**: Yes/No answers and extracted entities

### Language Classification
- **Architecture**: XLM-RoBERTa + custom classifier
- **Dataset**: 90k samples across 20 languages
- **Preprocessing**: Text tokenization with 128-token limit
- **Metrics**: Weighted F1-score and accuracy

### Event Detection
- **Task**: Sequence labeling for temporal events
- **Architecture**: Custom neural network
- **Data Format**: JSONL with token-label pairs
- **Evaluation**: Sequence-level evaluation metrics

### SimLex Evaluation
- **Objective**: Evaluate word embeddings on semantic similarity
- **Dataset**: SimLex-999 benchmark
- **Methods**: Word2Vec with various parameters
- **Analysis**: Correlation with human similarity judgments

### Visual Word Sense Disambiguation
- **Method**: Contextual similarity between glosses and both literal and image contexts
- **Models**: CLIP for image-text alignment, BLIP for VQA
- **Evaluation**: Cross-modal semantic understanding
- **Languages**: Multilingual support via sentence transformers

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Colab account (for cloud execution)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/LorenzoCiarpa/MNLP.git
cd MNLP
```

2. **Install dependencies**:
```bash
pip install torch torchvision transformers
pip install pytorch-lightning
pip install nltk spacy scikit-learn
pip install sentence-transformers
pip install salesforce-lavis
pip install gensim seqeval
pip install matplotlib seaborn tqdm
pip install gdown googletrans==4.0.0-rc1
```

3. **Download spaCy models**:
```bash
python -m spacy download en_core_web_sm
```

4. **Download NLTK data**:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('omw-1.4')
```

## 🚀 Usage

### Running Individual Tasks

Each task is implemented in a separate Jupyter notebook. Open the desired notebook and run all cells:

1. **Named Entity Recognition**:
```bash
jupyter notebook EntityRecognition.ipynb
```

2. **Word Sense Disambiguation**:
```bash
jupyter notebook WordSenseDisambiguation.ipynb
```

3. **Question Answering**:
```bash
jupyter notebook QuestionAnswering.ipynb
```

4. **Language Classification**:
```bash
jupyter notebook LanguageClassification.ipynb
```

5. **Event Detection**:
```bash
jupyter notebook EventDetection.ipynb
```

6. **SimLex Evaluation**:
```bash
jupyter notebook SimLex.ipynb
```

7. **Visual WSD**:
```bash
jupyter notebook VisualWordSenseDisambiguation.ipynb
```

### Google Colab Execution

All notebooks are designed to run in Google Colab with automatic dataset downloads and dependency installation.

## 📚 Reports

Detailed reports are available for key tasks:

- **[Named Entity Recognition Report](reports/EntityRecognition_report.pdf)**: Comprehensive analysis of table-filling approach with multiple transformer models
- **[Word Sense Disambiguation Report](reports/WordSenseDisambiguation_report.pdf)**: In-depth evaluation of contextual similarity methods


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

*This project demonstrates advanced competency in multilingual natural language processing, covering the full spectrum from traditional NLP tasks to cutting-edge multi-modal applications.*