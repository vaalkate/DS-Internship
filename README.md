Задача:
Ранжирование товаров (item_id) внутри каждого запроса (query_id).
Метрика качества — NDCG@10.

Подход:
Используется классический learning-to-rank на CatBoostRanker (YetiRank),
без нейросетей и GPU, с упором на устойчивость и ограничение по памяти.

Данные:
- Train / Test в формате Parquet
- Большие текстовые поля (item_description) не загружаются целиком в RAM

Работа с памятью:
- item_description читается по строкам через pyarrow
- Извлекаются только агрегированные признаки:
  - desc_len  — длина описания
  - desc_words — количество слов
Это позволяет избежать переполнения ОЗУ в Google Colab.

Фичи:
1) Текстовые:
   - cos_q_title       — cosine similarity (query_text vs item_title)
     (HashingVectorizer, без словаря)
   - overlap_q_title   — доля слов запроса, найденных в заголовке

2) Категориальные:
   - cat_match   — совпадение категории
   - mcat_match  — совпадение подкатегории
   - loc_match   — совпадение региона

3) Числовые:
   - log_price
   - item_query_click_conv
   - desc_len
   - desc_words

Модель:
CatBoostRanker:
- loss_function = YetiRank
- eval_metric   = NDCG@10
- depth = 7
- learning_rate = 0.1
- iterations = 700
- ранжирование по group_id = query_id

Валидация:
- train / val split выполняется по уникальным query_id
- утечки между запросами отсутствуют
