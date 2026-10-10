# AIAP Module 5 Mini-Project 1

Dataset: place the raw data file in `data/` (not tracked by git).

## Structure

```
data/              raw dataset (gitignored)
eda.ipynb          exploratory data analysis
main.py            pipeline entry point
models/            saved model artifacts (gitignored)
reports/           metrics JSON and figures
src/               pipeline modules + config.yaml
tests/             pytest suite
```

## Setup

```bash
pip install -r requirements-dev.txt
python main.py
pytest
```

## Pipeline (`python main.py`)

Predicts `shares` (regression) from the 19 features chosen in `eda.ipynb`. All settings live in `src/config.yaml`.

1. **Validate** the config and the raw data; write `reports/data_validation_report.json`.
2. **Clean** (`src/data_preparation.py`, following the EDA):
   - `data_channel` NaN → `'other'`.
   - Missing video / image / link counts → 0.
   - Clip the −1 keyword placeholders at 0.
   - Add `is_no_text`, `pub_year` / `pub_month` (from the URL) and `is_weekend`.
   - Drop `ID` / `URL`.
3. **Split** 80/10/10 into train/validation/test; the target becomes `log1p(shares)`.
4. **Preprocess** inside each model pipeline: `log1p` + `StandardScaler` for the heavy-tailed counts, `StandardScaler` for the other numeric features, one-hot for `data_channel` / `weekday`. Unlisted columns (including `n_comments`, excluded as leakage) are dropped.
5. **Train**:
   - Baselines: dummy (mean) and Linear Regression.
   - Tuned with `GridSearchCV` (5-fold, RMSE): Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting.
6. **Select** the lowest validation RMSE, refit on train+val, and evaluate once on test. Compute permutation feature importance.
7. **Save**:
   - `models/best_pipeline.joblib`
   - `reports/*.json`
   - `reports/figures/*.png`

A full run takes about 7–8 minutes on 16 cores, mostly the Random Forest and Gradient Boosting grids.

## Results and discussion

### Model performance

All metrics are on the validation set unless stated, with the target on the `log1p(shares)` scale. "Median error" is the median absolute error converted back to shares.

| Model | RMSE | R² | Median error (shares) |
|---|---|---|---|
| **Gradient Boosting** (selected) | **0.857** | **0.158** | **734** |
| Random Forest | 0.858 | 0.156 | 740 |
| Ridge | 0.891 | 0.090 | 773 |
| Lasso | 0.891 | 0.089 | 772 |
| Linear Regression | 0.891 | 0.089 | 772 |
| Decision Tree | 0.892 | 0.088 | 780 |
| Dummy (always predicts the mean) | 0.934 | 0.000 | 853 |

On the **test set**, the selected Gradient Boosting model (refit on train + validation) scores **RMSE 0.852, R² 0.157**, with a median error of 715 shares. The mean-only baseline scores 840.

Every model beats the baseline, and the tree ensembles beat the linear models. That fits the non-linear relationships seen in the EDA (§5.2). Still, the best model explains only **~16% of the variance** in log-shares. An RMSE of 0.85 on the log scale means a typical prediction is off by a **factor of about 2.3**: an article predicted at 1,400 shares could just as easily get 600 or 3,300. The predicted-vs-actual plot (`reports/figures/predicted_vs_actual_test.png`) shows the predictions bunched around the average. The model cannot single out the viral articles, nor the ones that flop.

### Why accuracy is so low

1. **Popularity depends mostly on things that are not in the data.** Whether an article spreads depends on its actual topic and how newsworthy it is, the quality of the headline and image, what else is in the news that day, who shares it first, and how social-media feeds rank it. The dataset describes an article's *form* (word counts, links, images, channel, day). It does not describe its *substance* or the social process that spreads it.
2. **Sharing is a self-reinforcing, partly random process.** Early shares lead to more visibility and then more shares, so small, chance differences in early attention grow into very large differences in final counts. Two articles with near-identical features can end up at 500 and 50,000 shares. Experiments on cultural markets (e.g. Salganik, Dodds & Watts, *Science*, 2006) found that the success of the same items varied widely between independent "worlds" with this kind of social influence. Part of this variance is irreducible noise that no feature set can remove.
3. **The target is extremely heavy-tailed.** `shares` ranges from 4 to 843,300 (skew ≈ 34). The log transform makes the problem learnable, but the viral tail is still where the largest errors are, and viral hits are by definition the hardest cases to predict.
4. **Every feature has a weak link to shares.** No usable feature has a Spearman correlation above 0.26 with log-shares (EDA §6.1). Combining many weak signals gives a modest model, not a strong one.
5. **The strongest signal was excluded on purpose.** `n_comments` was the feature most correlated with shares (ρ ≈ 0.36). Comments, however, build up *after* publication, like shares, so using them would leak the outcome into the inputs. The model is restricted to information available at publication time, which is the honest but harder setting.
6. **This version of the dataset has no content-understanding features.** It has 30 columns. Richer versions of this Mashable data also include topic and sentiment measures of the text. Without them, the model cannot tell an article on a hot topic from one on a dull topic of the same length.

### Feature categories: which kind of information predicts popularity?

The 19 model features fall into three groups:

| Category | What it describes | Features |
|---|---|---|
| **Textual** | The words of the article itself | `n_tokens_title`, `n_tokens_content`, `n_non_stop_unique_tokens`, `average_token_length`, `is_no_text` |
| **Structural** | How the page is built: links and media | `num_hrefs`, `num_self_hrefs`, `num_imgs`, `num_videos` |
| **Metadata** | Context around the article: keyword history, the popularity of linked articles, channel, publication timing | `kw_avg_min`, `kw_avg_max`, `kw_avg_avg`, `self_reference_avg_shares`, `num_keywords`, `data_channel`, `weekday`, `is_weekend`, `pub_month`, `timedelta` |

`self_reference_avg_shares` comes from links, but it measures the *past popularity* of the linked articles rather than the page's structure, so it is grouped as metadata.

Permutation importance on the test set is the increase in RMSE when a feature is shuffled (`reports/feature_importance.json`). Summed by category:

| Category | Features | Total importance | Share of total | Per feature | Strongest member (importance / EDA Spearman ρ) |
|---|---|---|---|---|---|
| **Metadata** | 10 | 0.113 | **83%** | 0.011 | `kw_avg_avg` (0.043 / 0.26) |
| Structural | 4 | 0.012 | 9% | 0.003 | `num_imgs` (0.005 / 0.07) |
| Textual | 5 | 0.011 | 8% | 0.002 | `n_tokens_content` (0.007 / 0.01) |

*Per-feature importances: metadata: `kw_avg_avg` 0.043, `data_channel` 0.023, `self_reference_avg_shares` 0.019, `is_weekend` 0.010, `timedelta` 0.008, `pub_month` 0.004, `kw_avg_max` 0.003, `kw_avg_min` 0.002, `num_keywords` 0.001, `weekday` 0.001. Structural: `num_imgs` 0.005, `num_self_hrefs` 0.004, `num_hrefs` 0.003, `num_videos` 0.001. Textual: `n_tokens_content` 0.007, `average_token_length` 0.002, `n_non_stop_unique_tokens` 0.001, `n_tokens_title` 0.000, `is_no_text` 0.000.*

**Metadata is clearly the most useful category.** It carries about 83% of the model's importance, and it stays ahead even after allowing for having the most features: 0.011 per feature, against 0.003 for structural and 0.002 for textual. The four strongest predictors are all metadata:
- **Keyword track record** (`kw_avg_avg`): articles on keywords that did well before tend to do well again.
- **Channel** (`data_channel`): social-media, tech and uncategorised blog posts are shared more; world news and entertainment less.
- **Popularity of linked articles** (`self_reference_avg_shares`).
- **Weekend publication** (`is_weekend`): less competition for readers' attention.

The common thread is that **past popularity of related content and the audience context predict future popularity better than anything about the article's own text or layout.**

Structural and textual features add only a little. More links and images help slightly, and longer articles do slightly better. Title length and the no-text flag contribute essentially nothing. This is consistent with point 6 above: the textual features here only *count* words, without capturing what the words are about.

Permutation importance is not strictly additive when features are correlated, so the category totals are approximate. The ranking (metadata ≫ structural ≈ textual) is clear, however, and it matches the EDA correlations.

### Possible next steps

- **Add content features:** topic modelling or sentence embeddings of the title and text, and sentiment and subjectivity scores. These are the most likely route to a better model at publication time.
- **Use early engagement:** if a prediction made shortly *after* publishing is acceptable, shares or comments in the first hour would be far stronger predictors (the `n_comments` result hints at this).
- **Reframe the target:** predicting "popular vs not popular" (e.g. above or below the median) or a range of likely shares (quantile regression) better reflects how uncertain a single exact number is.
