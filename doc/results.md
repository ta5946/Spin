# Development set

## Base models

| Group                      | Model                                | Threshold | Auc score | Accuracy score | Precision | Recall | F1 score  |
|----------------------------|--------------------------------------|-----------|-----------|----------------|-----------|--------|-----------|
| Baseline models            | Constant classifier                  | -         | 0.500     | 0.785          | 0.000     | 0.000  | 0.000     |
| Baseline models            | _Random classifier_                  | 0.2       | 0.586     | 0.370          | 0.239     | 0.884  | _0.376_   |
| Lexical measures           | Lemmas similarity                    | 0.3       | 0.836     | 0.840          | 0.593     | 0.814  | 0.686     |
| Lexical measures           | **Stems similarity**                 | 0.3       | 0.876     | 0.835          | 0.576     | 0.884  | **0.697** |
| String measures            | Levenshtein distance                 | 0.3       | 0.799     | 0.855          | 0.750     | 0.488  | 0.592     |
| String measures            | **Sequence similarity**              | 0.4       | 0.849     | 0.870          | 0.743     | 0.605  | **0.667** |
| Vector based measures      | Spacy embeddings                     | 0.6       | 0.710     | 0.710          | 0.388     | 0.605  | 0.473     |
| Vector based measures      | **Word2Vec embeddings**              | 0.5       | 0.815     | 0.800          | 0.527     | 0.674  | **0.592** |
| Ontology based measures    | **Path similarity**                  | 0.3       | 0.748     | 0.810          | 0.553     | 0.605  | **0.578** |
| Ontology based measures    | Leacock Chodorow similarity          | 1.6       | 0.734     | 0.825          | 0.595     | 0.581  | 0.588     |
| Ontology based measures    | Wu Palmer similarity                 | 0.4       | 0.737     | 0.765          | 0.466     | 0.628  | 0.535     |
| Pretrained language models | SciBERT embeddings                   | 0.8       | 0.778     | 0.760          | 0.458     | 0.628  | 0.529     |
| Pretrained language models | BioBERT embeddings                   | 0.9       | 0.823     | 0.780          | 0.492     | 0.698  | 0.577     |
| Pretrained language models | **Sentence transformers embeddings** | 0.4       | 0.877     | 0.805          | 0.532     | 0.767  | **0.629** |


## Large language models

### Text

| Model | Prompt                        | Auc score | Accuracy score | Precision | Recall | F1 score |
|-------|-------------------------------|-----------|----------------|-----------|--------|----------|
| OLMo  | sentence_template             | 0.801     | 0.860          | 0.667     | 0.698  | 0.682    |
| OLMo  | outcome_template              | 0.868     | 0.820          | 0.547     | 0.953  | 0.695    |
| OLMo  | role_template                 | 0.833     | 0.870          | 0.673     | 0.767  | 0.717    |
| OLMo  | article_definition_template   | 0.818     | 0.860          | 0.653     | 0.744  | 0.696    |
| OLMo  | wikipedia_definition_template | 0.801     | 0.860          | 0.667     | 0.698  | 0.682    |
| OLMo  | chain_of_thought_template     | 0.568     | 0.785          | 0.500     | 0.186  | 0.271    |
| OLMo  | random_example_template       | 0.697     | 0.630          | 0.347     | 0.814  | 0.486    |
| OLMo  | similar_example_template      | 0.717     | 0.635          | 0.356     | 0.860  | 0.503    |


### Probability

TODO Prompt selection

| Model      | Prompt          | Threshold | Auc score | Accuracy score | Precision | Recall    | F1 score |
|------------|-----------------|-----------|-----------|----------------|-----------|-----------|----------|
| OLMo       | role_template   | 0.1       | 0.914     | 0.870          | 0.667     | 0.791     | 0.723    |
| Mistral    | role_template   | 0.8       | 0.912     | 0.815          | 0.548     | 0.791     | 0.648    | 
| BioMistral | role_template   | 0.2       | 0.922     | 0.845          | 0.611     | 0.767     | 0.680    |
| Llama      | role_template   | 0.1       | 0.927     | 0.885          | 0.794     | 0.628     | 0.701    |
| OLMo       | detail_template | 0.3       | 0.908     | 0.880          | 0.721     | 0.721     | 0.721    |
| Mistral    | detail_template | 0.4       | 0.935     | 0.890          | 0.784     | 0.674     | 0.725    |
| BioMistral | detail_template | 0.1       | 0.924     | 0.825          | 0.561     | 0.860     | 0.679    |
| Llama      | detail_template | 0.1       | 0.935     | 0.830          | 0.846     | 0.256     | 0.393    |
| ChatGPT    | role_template   | 0.1       | 0.932     | 0.855          | 0.850     | 0.395     | 0.540    |


## Hard voting models

| Models                                                                      | Threshold | Auc score | Accuracy score | Precision | Recall | F1 score  |
|-----------------------------------------------------------------------------|-----------|-----------|----------------|-----------|--------|-----------|
| **Stems similarity, Sequence similarity, Sentence transformers embeddings** | 2         | 0.853     | 0.875          | 0.673     | 0.814  | **0.737** |
| **Stems similarity, Path similarity, Sentence transformers embeddings**     | 2         | 0.875     | 0.870          | 0.644     | 0.884  | **0.745** |
| OLMo, Mistral, BioMistral (role_template prompt)                            | 2         | 0.852     | 0.860          | 0.632     | 0.827  | 0.720     |
| OLMo, BioMistral, Llama (rol_template prompt)                               | 2         | 0.829     | 0.890          | 0.756     | 0.721  | 0.738     |
| **OLMo, Mistral, BioMistral (detail_template prompt)**                      | 2         | 0.887     | 0.915          | 0.783     | 0.837  | **0.809** |



# Test set

## Base models 

| Model                                | Threshold | Auc score | Accuracy score | Precision | Recall | F1 score  | Evaluation time |
|--------------------------------------|-----------|-----------|----------------|-----------|--------|-----------|-----------------|
| **Stems similarity**                 | 0.3       | 0.866     | 0.805          | 0.528     | 0.845  | **0.650** | 0.7 s           |
| Sequence similarity                  | 0.4       | 0.808     | 0.845          | 0.668     | 0.550  | 0.603     | 0.4 s           |    
| Word2Vec embeddings                  | 0.4       | 0.820     | 0.708          | 0.409     | 0.813  | 0.544     | 1.5 s           |
| Path similarity                      | 0.2       | 0.760     | 0.795          | 0.517     | 0.673  | 0.585     | 6.7 s           |
| **Sentence transformers embeddings** | 0.4       | 0.902     | 0.821          | 0.553     | 0.862  | **0.673** | 45.6 s          |


## Large language models

| Model          | Prompt              | Threshold | Auc score | Accuracy score | Precision | Recall | F1 score  | Evaluation time |
|----------------|---------------------|-----------|-----------|----------------|-----------|--------|-----------|-----------------|
| OLMo           | role_template       | 0.1       | 0.922     | 0.866          | 0.653     | 0.802  | 0.720     | 816.8 s         |
| Mistral        | role_template       | 0.3       | 0.912     | 0.852          | 0.625     | 0.774  | 0.692     | 910.1 s         |
| **BioMistral** | **role_template**   | 0.2       | 0.925     | 0.881          | 0.712     | 0.751  | **0.731** | 910.5 s         |
| Llama          | role_template       | 0.1       | 0.918     | 0.884          | 0.831     | 0.578  | 0.681     | 930.1 s         |
| **OLMo**       | **detail_template** | 0.3       | 0.931     | 0.889          | 0.725     | 0.778  | **0.750** | 1120.6 s        |
| Mistral        | detail_template     | 0.2       | 0.932     | 0.894          | 0.789     | 0.692  | 0.737     | 1220.9 s        |
| BioMistral     | detail_template     | 0.1       | 0.926     | 0.851          | 0.609     | 0.852  | 0.710     | 1222.0 s        |
| Llama          | detail_template     | 0.1       | 0.945     | 0.853          | 0.962     | 0.329  | 0.490     | 1127.4 s        |


## Hard voting models

| Models                                                                      | Threshold | Auc score | Accuracy score | Precision | Recall | F1 score  |
|-----------------------------------------------------------------------------|-----------|-----------|----------------|-----------|--------|-----------|
| **Stems similarity, Sequence similarity, Sentence transformers embeddings** | 2         | 0.838     | 0.854          | 0.622     | 0.811  | **0.704** |
| **Stems similarity, Path similarity, Sentence transformers embeddings**     | 2         | 0.833     | 0.832          | 0.575     | 0.834  | **0.680** |
| OLMo, Mistral, BioMistral (role_template prompt)                            | 2         | 0.845     | 0.877          | 0.683     | 0.791  | 0.733     |
| OLMo, BioMistral, Llama (role_template prompt)                              | 2         | 0.838     | 0.899          | 0.781     | 0.733  | 0.756     |
| **OLMo, Mistral, BioMistral (detail_template prompt)**                      | 2         | 0.862     | 0.901          | 0.758     | 0.793  | **0.775** |


## Fine tuned models (reported)

| Model             | Auc score | Accuracy score | Precision | Recall | F1 score  |
|-------------------|-----------|----------------|-----------|--------|-----------|
| BERT uncased      | -         | -              | 0.858     | 0.882  | 0.868     |
| SciBERT uncased   | -         | -              | 0.880     | 0.908  | 0.893     |
| **BioBERT cased** | -         | -              | 0.889     | 0.908  | **0.898** |
