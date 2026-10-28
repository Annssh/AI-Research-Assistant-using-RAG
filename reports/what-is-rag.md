# What is RAG (Retrieval‑Augmented Generation)

## Summary
Retrieval‑Augmented Generation (RAG) augments a pretrained generative language model by retrieving relevant external documents at query time and conditioning the generator on those texts, improving factuality and recency compared with a standalone model [6]. The approach was introduced in the 2020 work by Lewis et al. and has been evaluated on open‑domain question answering, summarization and fact verification tasks [1]. Typical RAG systems index a document collection, embed and store document chunks in a non‑parametric retriever, fetch top‑k passages for each query, and feed those passages into a sequence‑to‑sequence generator to produce an answer [6]. Variants and practical production patterns (vector DBs, embedding services, hybrid retrievers) are commonly described in implementation guides and architecture posts [4][6].

## Key Findings

### Definition and origins
RAG is a framework that couples a retrieval component (non‑parametric memory) with a parametric generator so model outputs can be grounded in retrieved text at inference time. The design aims to enable more factual and up‑to‑date responses without requiring the generator to encode all knowledge in its parameters [6]. The technique traces to the Lewis et al. 2020 paper and associated implementations summarized in practitioner writeups [1].

### Core pipeline and components
A canonical RAG pipeline consists of four steps: (1) index/source documents (usually chunked into passages), (2) embed or encode those chunks and store them in a vector/non‑parametric retriever, (3) retrieve the top‑k passages relevant to an input query, and (4) condition a seq2seq generator on the retrieved passages to produce the final output [6]. Implementations vary in retriever type (dense vs sparse), chunking strategy, and how the generator consumes retrieved context [6][4].

![RAG end-to-end architecture](https://nlp.cs.ucl.ac.uk/publications/2020-05-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks/featured.jpg)
*Figure 1: End-to-end RAG architecture showing query encoding, non‑parametric retrieval, and generation. (Image source: nlp.cs.ucl.ac.uk)*

### Implementation choices and training
The original RAG work proposed dense retrievers such as Dense Passage Retrieval (DPR) paired with a sequence‑to‑sequence generator (e.g., BART). In practice, retriever and generator can be fine‑tuned jointly in the RAG setup, while the document encoder may be kept fixed to avoid frequent index rebuilds [1]. Production systems commonly integrate vector databases and embedding services to manage scale and latency [4][6].

### RAG variants: Sequence vs Token and architecture patterns
Two principal decoding variants are commonly discussed: RAG‑Sequence, which conditions a full generated hypothesis on a single retrieved document and thus preserves document‑level consistency; and RAG‑Token, which marginalizes over retrieved documents at the token level, allowing the generator to synthesize information across multiple sources per token [8]. Beyond these, many RAG design variants exist (branched/adaptive retrieval, memory extensions, corrective loops) to address consistency, latency, and updateability in deployed systems [6][4].

![Parametric vs non-parametric vs RAG comparison](https://media.springernature.com/lw1200/springer-static/image/art%3A10.1007%2Fs12599-025-00945-3/MediaObjects/12599_2025_945_Fig1_HTML.png)
*Figure 2: Comparison of parametric memory, non‑parametric memory, and RAG pipelines. (Image source: media.springernature.com)*

### Practical benefits and use cases
RAG reduces hallucination risk by grounding generations in retrieved passages and allows systems to leverage external corpora without full model re‑training. It is especially applicable to knowledge‑intensive tasks such as open‑domain QA, customer support, fact verification, and document‑grounded generation [5][6].

## Conclusion
RAG is a pragmatic architecture pattern that combines retrieval over an external corpus with conditional generation to produce more factual and up‑to‑date outputs. The core pipeline—index, embed, retrieve, generate—supports multiple implementation choices (dense/sparse retrievers, joint fine‑tuning, RAG‑Sequence vs RAG‑Token) and a range of production variants. For knowledge‑intensive applications where factual grounding and updatability matter, RAG provides a flexible way to extend a generative model’s capabilities without embedding all knowledge parametrically [1][6][5].

## References
1. https://latenode.com/news/rag-lewis-2020-paper-understanding-original-retrieval-augmented-generation-research
2. https://www.scirp.org/reference/referencespapers?referenceid=3915330
3. https://medium.com/@AriaLeeNotAriel/numbynum-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-lewis-et-al-df93a0f4c8f0
4. https://latenode.com/blog/rag-architecture-diagram
5. https://www.hebbia.com/resources/rag-architecture
6. https://humanloop.com/blog/rag-architectures
7. https://cameronrwolfe.substack.com/p/a-practitioners-guide-to-retrieval
8. https://www.comet.com/site/blog/retrieval-augmented-generation
9. https://www.exxactcorp.com/blog/deep-learning/how-retrieval-augment-generation-makes-llms-smarter-than-before