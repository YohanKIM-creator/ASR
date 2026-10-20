# Распознавание речи: Conformer CTC

Распознавание английской речи на LibriSpeech. За основу взят [учебный шаблон](https://github.com/Blinorot/pytorch_project_template/tree/example/asr). В проекте реализованы Conformer, CTC prefix beam search, четыре аудиоаугментации и декодирование с KenLM.

Модель wide288 шага 76 000 выбрана по dev-other. В таблице — результаты на полных test-clean и test-other. [Отчёт с графиками](https://www.comet.com/kimigor157/hse-asr-conformer/00b40d3352ed4b02ab6e7c3854d83ca9?experiment-tab=html), [счётчики ошибок](https://www.comet.com/kimigor157/hse-asr-conformer/52af7a4dc1a141568aa0768e078f7ead?experiment-tab=html).

<table>
<thead>
<tr><th rowspan="2">Inference type</th><th colspan="2">WER, %</th><th colspan="2">CER, %</th></tr>
<tr><th>test-clean</th><th>test-other</th><th>test-clean</th><th>test-other</th></tr>
</thead>
<tbody>
<tr><td>CTCArgmax</td><td>9.01</td><td>20.66</td><td>2.69</td><td>7.86</td></tr>
<tr><td>Beam Search (beam_size=16)</td><td>8.81</td><td>20.38</td><td>2.65</td><td>7.75</td></tr>
<tr><td>LM (beam_size=32)</td><td>5.44</td><td>14.16</td><td>1.98</td><td>6.59</td></tr>
</tbody>
</table>

LM: публичная 4-gram из OpenSLR, α=0,75, β=2,00. Prefix beam search использует `token_topk=8`, как в конфигурации инференса. Все три декодера оценены на одном CUDA FP32-кэше.

Checkpoint 81 500 с той же LM получил WER **5,29% / 13,93%** и CER **1,92% / 6,45%** на test-clean / test-other ([логи](https://www.comet.com/kimigor157/hse-asr-conformer/6d5ecc7162504c72bd6947c4ae265334?experiment-tab=html)).

## Установка

Нужны Linux, Python 3.12 и CUDA GPU. Команды установки приведены для Ubuntu. Все Python-пакеты указаны в [requirements.txt](requirements.txt).

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv build-essential

git clone https://github.com/YohanKIM-creator/ASR.git
cd ASR
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Тесты:

```bash
python -m pytest -q tests
```

## Веса и логи

Опубликованы компактные веса [шага 76 000](https://www.comet.com/api/asset/download?experimentKey=00b40d3352ed4b02ab6e7c3854d83ca9&assetId=lCgwCus9wAaLLBPoVlQRolVBR) и [шага 81 500](https://www.comet.com/api/asset/download?experimentKey=00b40d3352ed4b02ab6e7c3854d83ca9&assetId=ZQTRBkulO3vc5lZYynn83BEFM). Шаг 76 000 выбран по dev; 81 500 — checkpoint конца основного запуска. Для скачивания весов и 4-граммной LM:

```bash
python scripts/download_assets.py --checkpoint-step 76000 --output data/resources
```

Для последнего checkpoint укажите `--checkpoint-step 81500`. Загрузчик распаковывает LM в ARPA. Другие веса можно передать через `--checkpoint-url`; для повторной загрузки файлов используйте `--force`.

Компактные веса содержат модель, конфигурацию и алфавит для инференса.

Полный журнал wide288, включая шаги 1–81 541, скачивается без API-ключа:

```bash
python scripts/download_logs.py --output-dir outputs/training-logs
```

| Этап | Логи |
| --- | --- |
| char100 | [Comet ML](https://www.comet.com/kimigor157/hse-asr-conformer/98a01799854f415c9c62067050863008?experiment-tab=html) |
| char960 | [Comet ML](https://www.comet.com/kimigor157/hse-asr-conformer/d307dfcc5a504fc4ad4afac43338b698?experiment-tab=html) |
| wide288 | [Comet ML](https://www.comet.com/kimigor157/hse-asr-conformer/00b40d3352ed4b02ab6e7c3854d83ca9?experiment-tab=panels) |

## Инференс

Аудио должно лежать в `dataset/audio/`, эталонные тексты — в необязательном каталоге `dataset/transcriptions/`. Например, `audio/sample.wav` и `transcriptions/sample.txt`. Поддерживаются WAV, FLAC и MP3; звук приводится к моно 16 кГц.

```bash
python inference.py \
  datasets.custom.data_dir=/path/to/dataset \
  inferencer.device=cuda \
  inferencer.from_pretrained=/path/to/wide288.pt \
  inferencer.save_path=outputs/predictions \
  inferencer.decoder=lm inferencer.lm_path=/path/to/4-gram.arpa \
  inferencer.lm_uppercase=true inferencer.lm_alpha=0.75 \
  inferencer.lm_beta=2.00 inferencer.beam_size=32
python scripts/calc_metrics.py \
  --reference-dir /path/to/dataset/transcriptions \
  --prediction-dir outputs/predictions/custom --output outputs/metrics.json
```

Для prefix beam search задайте `inferencer.decoder=beam inferencer.beam_size=16 inferencer.token_topk=8`, для argmax — `inferencer.decoder=greedy`. Оба режима работают без LM. Метрики считаются по суммарному числу ошибок во всём наборе; ID предсказаний и эталонов должны совпадать. Для оценки test-clean/test-other используйте `datasets=evaluation` с теми же весами и параметрами декодера.

## Demo

[Ноутбук](notebooks/inference_colab.ipynb) устанавливает зависимости, загружает модель, распознаёт три примера и архив по `DATASET_URL`. Выберите GPU runtime и checkpoint: `76000` или `81500`. Demo проверен с опубликованными весами на трёх встроенных записях и том же архиве, скачанном по URL.

Архив можно скачать отдельно: `python scripts/download_dataset.py --url URL --output data/custom`.

Три примера из [LibriSpeech dev-clean](https://www.openslr.org/12/) лежат в [demo/librispeech-sample.zip](demo/librispeech-sample.zip).

## Воспроизведение обучения

Модель обучалась в три этапа: char100 — 46 000 шагов, char960 — 22 000, затем расширение до wide288 и обучение до шага 81 500. По dev-other выбран checkpoint 76 000; он же использован в основной таблице test-результатов. Для повторения на GPU:

```bash
python scripts/prepare_librispeech.py \
  --parts train-clean-100 train-clean-360 train-other-500 dev-clean dev-other
git checkout train-char100
python train.py --config-name train_char100
git checkout train-char960
python train.py --config-name train_char960
git checkout train-wide288
python scripts/widen_conformer.py --device cuda \
  --checkpoint saved/char960/last.pt --output saved/wide288/initialization.pt
python train.py --config-name train_wide288 trainer.session_max_steps=81500
git checkout main
```

Параметры этапов: [char100](src/configs/train_char100.yaml), [char960](src/configs/train_char960.yaml), [wide288](src/configs/train_wide288.yaml). Первые два тега используют W&B (`WANDB_API_KEY`), wide288 — Comet (`COMET_API_KEY`). Для запуска без трекера добавьте `writer.mode=disabled`.

`init_from` переносит веса в новый этап. `resume_from=auto` продолжает прерванный запуск с optimizer, scheduler, scaler, RNG и позицией в данных. `session_max_steps` ограничивает текущую сессию, а длина расписания LR задаётся отдельно.

Снимки кода этапов обучения доступны в тегах [train-char100](https://github.com/YohanKIM-creator/ASR/tree/train-char100), [train-char960](https://github.com/YohanKIM-creator/ASR/tree/train-char960) и [train-wide288](https://github.com/YohanKIM-creator/ASR/tree/train-wide288).
