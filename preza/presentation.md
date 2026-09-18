---
marp: true
theme: graph_paper
paginate: true
header: ООО «Нетология»
footer: Сентябрь, 2026г.
---

### Разработка гибридной модели для предсказания следующего элемента в пользовательской сессии на основе поведенческих паттернов и контентных эмбеддингов видео.

###### Сравнение четырёх классов рекомендательных моделей на датасете VK-LSVD

<style scoped>
.columns {
    display: grid;
    grid-template-columns: 40% 60%;
    gap: 1rem;
    margin-top: 60px;
    justify-items: start;
}
.logos {
    justify-self: center;
}
.authors {
    padding-top: 100px;
    color: #575279;
}
.add-info {
    text-align: center;
    font-size: 16pt;
    color: #575279;
}
header, footer {
    left: 0;
    right: 0;
    text-align: center;
}
</style>

<div class="add-info">Итоговый проект по модулю "Машинное обучение: фундаментальные инструменты и практики"</div>

<div class="columns">
<div class="logos">

![w:250px](netology.png)

</div>
<div class="authors">
Автор: Игорь Каукин, группа: AML-68<br>
Куратор: Максим Зайкин<br>
Ментор: Екатерина Прохорова
</div>
</div>

---

# Контекст задачи

**Рекомендательные системы** — ключевой инструмент современных видеоплатформ.

- Пользователь не формулирует запрос — система **проактивно** предлагает контент
- Основа — **история взаимодействий**: просмотры, реакции, время просмотра
- Цель — максимизировать вероятность **целевого действия**

**Задача:** предсказать, какой ролик пользователь посмотрит **следующим**.

---

# Постановка задачи

**Дано:**
- $\mathcal{U}$ — пользователи, $\mathcal{I}$ — видеоролики
- $H_u = [i_1, i_2, \ldots, i_t]$ — история взаимодействий пользователя $u$

**Требуется:** для каждого $u$ построить функцию скоринга

$$f_u: \mathcal{I} \setminus H_u \rightarrow \mathbb{R}$$

и вернуть топ-$K$ роликов:

$$\text{top-}K(u) = \underset{i \in \mathcal{I} \setminus H_u}{\text{argtop-}K} \; f_u(i)$$

**Метрики:** Recall@K, NDCG@K, MRR@K

---

# Данные: VK-LSVD

<!-- _class: tinytext -->

**VK-LSVD** — открытый промышленный датасет коротких видео (AI VK Research).

- Полная версия: **40+ млрд** взаимодействий, 10M пользователей, 20M видео
- В работе: **подвыборка** `up0.001_ip0.001`
- **25 недель**: train — 0..24, validation — 25
- ~600 тыс. взаимодействий/неделю, ~2.6 ГБ

**Разметка pos/neg:**
- **Позитив:** `timespent >= 5` сек **или** любая реакция, **при отсутствии** `dislike`
- **Негатив:** всё остальное

**Особенность:** нет timestamp — хронология по порядку строк в parquet.

---

# Четыре модели — сравнение

| Модель | Класс | Что учитывает |
|---|---|---|
| **MostPopular** | неперсональный | глобальная популярность |
| **FPMC** | марковский | последний item + вкус пользователя |
| **LightGCN** | графовый | коллаборативные связи |
| **SASRec** | sequential | всю историю через self-attention |

**Гипотеза:** разные классы моделей предпочтительны для **разных групп пользователей** по длине истории.

---

# Архитектура FPMC

*Factorized Personalized Markov Chains*

$$\text{score}(u, i, j) = \underbrace{p_u^\top q_j}_{\text{MF}} + \underbrace{\sum_d BU_{u,d} \, T_{i,d} \, T_{j,d}}_{\text{MC}}$$

- **MF** — персональные предпочтения
- **MC** — переходы «последний → следующий» с учётом пользователя

**Обучение:** BPR-loss, negative sampling

**Особенность:** учитывает **только последний** item.

---

# Архитектура SASRec

*Self-Attentive Sequential Recommendation*

[//]: # (<img src="https://mermaid.ink/img/pako:eNpFkkFu2zAQRa9CTJaRVFG0VYkIAiSpjRiw3SzSTaUsGHNsE5FIg6SauIYXzSIHyDqny0kqs7DLBTGD_98fgpgdLIxE4LBszPNiLawn99e1rTXpz6Sa6E3nOVGfr3-iw_0akSRJQvnxQOL4koyqiceWnJM745RXRpNR-4hSKr1yD8egUbBO59VUbNHOjW1P0nQetNntVTXrGq_iWxSSXHmP-pB28Wi_XJ6TG9E50ZCZcE8nskcCOh7PqzGijMfGPgsrT4ZeCIbvP-77wc6TzfGJn2_vpHNoyS9ceGNPhPPbBkPwUjUNP1uWosiGkfPWPCE_Y4xBBCurJHBvO4ygRduKQwu7Q0INfo0t1sD70qLsXuKFaYytodb7Ht0I_dOY9khb063WwJeicX3XbaTw-E2JlRX_Lagl2hvTaQ-c0rIIIcB38AI8LxLKhowWgzSlrMxYBFvg5SBhOaWUFYx-HeY030fwO0xNkzxnA5Zl2bBIs3RAywhQqv4HZv-WIOzC_i-H0Kce?type=png" height="80%">)
<img src="https://mermaid.ink/img/pako:eNo9ksFu2zAMhl9FYI-1vUR2ElsoChRdggZI0mHYLrN7UCMmMWpbAS2tTYMc2sMeYOc9XZ9ksopYB4kU_--nIPAIa60QBGwq_bzeSTJs8Z25VVDRdOc8nzd7awQrP97fgm5_D1gURT7898DC8JpN87nBml2yb7otTakbNq0fUamy2bYPZ6Oply5W-UIekFaa6r60WPna8u4mX9rKlOEdSsVujMGmc7t6pC_Xl-xW2lZWbCnbp550iEdns1U-Q1ThTNOzJNULXMEL7n_-cI1bw_bnJ378-ctsi8R-49po6onWHCr0xpuyqsTFJpMpHwWtIf2E4iKOYwhgS6UCYchiADVSLbsUjp1DAWaHNRYgXEio7Eu41pWmAorm5NC9bH5pXZ9p0na7A7GRVesyu1fS4NdSbknW_S1ho5ButW0MiOHEe4A4wguIlEdJlo2GPI3TAU_GSQAHp4l5NEkm6ZjzmGfxgMenAF5910E04uMsHY6S4WSQpkkyDgBV6X5g-TkIfh5O_wHEVKfc?type=png)](https://mermaid.live/edit#pako:eNo9ksFu2zAMhl9FYI-1vShyPEcYChRdggZIsmHYLrN7UCMmMWpLASWtzYIc1sMeYOc9XZ9ktodYB4kU_--nBPAEG6sRJGxr-7zZK_Js-YW1q6TSdOeiWJhD8JJVb6-_om5_jViSJH3494HF8Q2bFQuPDbtmn62rfGUNmzWPqHVldu7hYjTrpct1sVRHpLWlZigt131tdX9brELtq_gelWa33qPp3D480ruba3anglM1Wyn3NJAt0qPz-bqYI-p4bulZkR4EbaEXfPr2tW3sPDtcnvj2-w8LDon9wI23NBDOH2vsjbdVXcur7VTl40nkPNknlFdCCIhgR5UG6SlgBA1So7oUTp1DCX6PDZYg25BQh5d4Y2tLJZTm3KIHZb5b21xosmG3B7lVtWuzcNDK48dK7Ug1wy2h0Uh3NhgPkqe9B8gTvIDM8oSLieB5OhpxMR2LCI4gp2kiMs65yAV_P8l4do7gZ990lGSZSMV4nI7SPOPpJALUVfv_1f8x6Kfh_A90EadU">

---

# Архитектура LightGCN

*Light Graph Convolutional Networks*

![w:600px center](https://mermaid.ink/img/pako:eNqVkk9LwzAYxr9KeU8rtKNJ_5qDIApe9OLBg9ZDXbM1sDYja3A6Bnrw5KcQv4KDDZmfIftGpiuto-DAHJK8T573l_CQOQx4SoHAcMwfBlkiSuPiKhZxYegxlfcjkUwyIwb1rr7Ut1objtFTq-3L9lVXn2qzfVNLM4amoxrS6fWkaRq2bRtM75lpNse0SA-xUQeE9kDoPyDcAeE9EP4LJB1tONa31gtudFbrrNYZ7vhPrs9v1YdO5Fktq0T0vFErtbxrfajxtQruKqxl4cMw1sJ-39fSMFgwEiwFUgpJLcipyJOqhHlljaHMaE5jIHoraCpn9oCPudBhFQvdOkmKG87zpltwOcqADJPxVFdykiYlPWOJDjtvVaHzo-KUy6IEglAU7ChA5jAD4rp9zw9cpPXQdyOELXgE4kd9JwjcEHl-6CEP-8HCgqfdvU4_cn2M3Cj0nBAdBZHuoCkrubisP-nury5-AJWb4t4?type=png)

---

# Гибридный ансамбль

<!-- _class: tinytext -->

[![center](https://mermaid.ink/img/pako:eNpdksFq20AQhl9lGV9lI2Ul2dpDIchtHbDrIrsEapkirJUtKmnDekWTGkOaQg49htzbS-k1lJq6pfUzrF6hT9BH6GqDDc2eZme-_58ZmBXMWEyBQJKxN7NFxAXqByEPC6Tei5PupFxS_iqNp6jZfIR6J6PxRH6Sd_K7_CU31SWS2-pddSV31aXcyi2SX1F1peKN_CF_y81071QLtUOwUvJd9V5plHy9rwe66A_73cnfj18-IJ9lMfpzfYOePB_40_-p0-NgoKjbz-g04rmmRsejgM4ecL3hWGMa7TGhyX46X4in_rMDW_e8n-yxX6-2kT_lrl5Ozf9N3lXX9V4Huu69pw-7DccPU0txkVE1RpJmGWlQz6IuNpaCs9eUNDDGxoxljJNGkiRgwJynMRDBS2pATnke1V9Y1VYhiAXNaQhEhZzG5XlTS0MIi7WSnkXFS8byvZqzcr4AkkTZUv3KszgStJtGcx7lhyynRUy5z8pCALHdI20CZAXnQJqWabc8z7Sw28Gu7bY7jgEXQCzbbGHsOEeO1XGsdsfEawPe6sZWy2o7uON5todd07baBtA4FYwP7u9Kn9f6H1DT3jM?type=png)](https://mermaid.live/edit#pako:eNpdks9q20AQxl9lGV9lY2nlP9pDIMhtHbDrIrsEYpkgrJUtKmnDekWTGkOaQg49lt7bS8g1hJi4pfUzrF6hT9BH6GqDDc2eZmd-3zczMEuYspACgShh76fzgAvU83zuZ0i9t0edcb6g_DQOJ6haPUDdo-FoLL_LO_kof8l1cYnkpvhYXMltcSk3coPkPSquVLyWP-RvuZ7snEqhdvCWSr4tPimNkq92dU8X3UGvM_777fYzclkSoj_XX9DLN3138j91fOj1FfX1Bh0HPNXU8HDo0ekzrjsYaUyjXSY02Ytnc_HKfb1ny55Pk71wy9XW8qfclsup-R_kXXFd7rWny947er_bYPQ8tRAXCVVjRHGSkAp1TNrExkJw9o6SCsbYmLKEcVKJoggMmPE4BCJ4Tg1IKU-D8gvL0soHMacp9YGokNMwP69qqQ9-tlLSsyA7YSzdqTnLZ3MgUZAs1C8_CwNBO3Ew40G6z3KahZS7LM8EENy0tAmQJZwDse1ave60HZW2WrjpmAZcADGxXcO40cBWs-3YVt3CKwM-6LZmzWw1cNtxbKWp22bLABrGgvH-01Xp41r9A56J3dA)

**Идея:** разные модели — для разных пользователей.

- **cold** (короткая история) → FPMC
- **warm** (средняя) → MostPopular
- **hot** (длинная) → SASRec

---

# Пайплайн обучения

[//]: # (<img src="https://mermaid.ink/img/pako:eNpdU9tu2kAQ_ZXVPhuKsTFgVZESCKRSqFCQ-lATVQ5eLpKx6WK3SRESIW2liodUlfrabyAltBQI_MLsH3W8VmiDJcs7nnPOzJnRDmjDdxg1adP13zfaNg_I6Vmd1z2Cz6FVtfnbkAVEfIQp_IGVmJyTROKAHFnwHe5hAQ8wFbdEjDD9G9YwF2NYwvT5BX924NoXzO0ne1fnj3pHkluw4AfciS_InhPYIncOG_wRa2zgJ34fYCFuTSkji9wjZimuxUQhfcY8hYhPiN9gZgMrfGcRflenIOsUVavi94Oq3wtdm-8n01apWinICmKMWgvYohL2jxb3oJpVO6ydsUYMvkZYNIvY_SyB1ZfRGPZZunXaabWDcuFlbGONk5KFxGeYEpTBacGvyAH6GsNixy-qUuDYgq9ihJm5HP4y6owke8E_XDrG7WJtL9afxscyLFnsne2GdsDetLgf9p4sqCQhZatQe0XibcrJLHFNsCEww35u8LjFftY7UlmSTiz4JiZwh3ZGEXcjd7aCBYmG8x9X3OyYJ5L5wup4TcaZ12BRL1ShLd5xqBnwkCm0y3jXjkI6iEh1GrRZl9WpiUfOnPAy0fBdn9dp3RsitWd7r32_-8hGh602NZu228co7Dnou9ixW9zu7v5iZYfxgh96ATXzekaKUHNAL6mZyyez6VQ6a6i6ljN0I6fQK2qqKT2Z1dO6mjFUQ81oRm6o0A-ybCqpZdWUkVdTmp7Na7mMrlDmdAKfV-J7Jq_b8C8ZM3Zk?type=png" width="85%">)
[![w:1200px center](https://mermaid.ink/img/pako:eNpdU9tu2kAQ_ZXVPhtqY5uLVUVKIJBKoUJB6kNNVDmwXCRj08VukyIkQtpKFQ-pKvW130BKaCkQ-IXZP-p4rdAGS5Z3POecmTOjHdC632DUok3Xf19vOzwgp2c1XvMIPod2xeFvQxYQ8RGm8AdWYnJOEokDcmTDd7iHBTzAVNwSMcL0b1jDXIxhCdPnF_zZgetcMLef7F2dP-odSW7ehh9wJ74ge05gi9w5bPBHrLGBn_h9gIW4taSMLHKPmKW4FhOF9BnzFCI-IX6DmQ2s8J1F-F2dvKxT0Oyy3w8qfi90Hb6fTNnFSjkvK4gxai1gi0rYP1rcg-p29bB6xuox-Bph0Sxi97MEVl9GY9hnGfZpp9UOSvmXsY01TkoWEp9hSlAGpwW_IgfoawyLHb-gSYFjG76KEWbmcvjLqDOS7AX_cKkYt4v1vdh4Gh_LsGizd44bOgF70-J-2HuyoKKElOx89RWJtykns8Q1wYbADPu5weMW-1nvSCVJOrHhm5jAHdoZRdyN3NkKFiQazn9ccbNjnkjmC7vjNRlnXp1FvVCFtninQa2Ah0yhXca7ThTSQUSq0aDNuqxGLTxy1ggvE3Xf9XmN1rwhUnuO99r3u49sdNhqU6vpuH2Mwl4DfRc6Tos73d1frNxgPO-HXkAtI52WItQa0EtqJTTVSOZyqqans3raSGeypkKvqKUZalLXTTNlallTy2RVfajQD7KwltQypp5TVdXUNC2dM3SFskYn8Hk5vmnywg3_Avdhdq0?type=png)](https://mermaid.live/edit#pako:eNpdksFq20AQhl9lGV9lI2Ul2dpDIchtHbDrIrsEapkirJUtKmnDekWTGkOaQg49htzbS-k1lJq6pfUzrF6hT9BH6GqDDc2eZme-_58ZmBXMWEyBQJKxN7NFxAXqByEPC6Tei5PupFxS_iqNp6jZfIR6J6PxRH6Sd_K7_CU31SWS2-pddSV31aXcyi2SX1F1peKN_CF_y81071QLtUOwUvJd9V5plHy9rwe66A_73cnfj18-IJ9lMfpzfYOePB_40_-p0-NgoKjbz-g04rmmRsejgM4ecL3hWGMa7TGhyX46X4in_rMDW_e8n-yxX6-2kT_lrl5Ozf9N3lXX9V4Huu69pw-7DccPU0txkVE1RpJmGWlQz6IuNpaCs9eUNDDGxoxljJNGkiRgwJynMRDBS2pATnke1V9Y1VYhiAXNaQhEhZzG5XlTS0MIi7WSnkXFS8byvZqzcr4AkkTZUv3KszgStJtGcx7lhyynRUy5z8pCALHdI20CZAXnQJqWabc8z7Sw28Gu7bY7jgEXQCzbbGHsOEeO1XGsdsfEawPe6sZWy2o7uON5todd07baBtA4FYwP7u9Kn9f6H1DT3jM)

---

# Результаты

<style scoped>
.columns {
    font-size: 12pt;
    display: grid;
    grid-template-columns: 40% 60%;
    gap: 1rem;
}
</style>

<div class="columns">
<div>
<p><b>Smoke-прогон:</b> 5 недель, 1 эпоха, подвыборка</p>

<p><i>Миссия этого прогона была только одна: сравнить модели на доступном автору железе.</i></p>
</div>
<div>

| Группа | Модель | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| cold | MostPopular | 0.0019 | 0.0014 | 0.0027 |
| cold | **FPMC** | **0.0062** | **0.0120** | 0.0101 |
| cold | LightGCN | 0.0047 | 0.0082 | 0.0014 |
| cold | SASRec | 0.0057 | 0.0061 | **0.0141** |
| warm | **MostPopular** | **0.0144** | **0.0141** | 0.0112 |
| warm | FPMC | 0.0032 | 0.0057 | 0.0008 |
| warm | LightGCN | 0.0040 | 0.0061 | 0.0019 |
| warm | SASRec | 0.0032 | 0.0048 | **0.0162** |
| hot | MostPopular | 0.0096 | 0.0233 | 0.0052 |
| hot | FPMC | 0.0018 | 0.0102 | 0.0039 |
| hot | LightGCN | 0.0039 | 0.0178 | 0.0022 |
| hot | **SASRec** | 0.0095 | **0.0237** | **0.0462** |

</div>
</div>

---

# Выводы

<!-- _class: tinytext -->

<style scoped>
li { font-size: 0.6rem; }
</style>

**Основные результаты:**

1. **FPMC** — лучший на **cold** (короткая история)
2. **MostPopular** — силён на **warm**
3. **SASRec** — лучший на **hot**, лидер по MRR
4. **LightGCN** — недообучен на 1 эпохе, требует больше

**Главное:** ни одна модель не выигрывает везде → **гипотеза гибридного ансамбля подтверждается**.

**Планы:**
- Адаптивная сегментация (границы под размер окна обучения).
- Fallback на обеих границах (когда размер истории не позволяет уверенно отнести пользователя к конкретной группе, можно
  предусмотреть специальную стратегию по добавлению прогноза от Most Popular модели).
- LightGCN на 10+ эпохах (модель не смогла проявить себя из за скромности использованных в опыте ресурсов).
- Проверка инсайтов найденных в EDA (результаты EDA в проекте не использовались, но ознакомится с ними можно в файле
  notebooks/01_eda.ipynb).
