# META-USAGE — використання ШІ в проєкті

Згідно з Методичними рекомендаціями (розділ 6) тут декларується весь згенерований код.

| Дата | Інструмент | Що зроблено | Файли |
|---|---|---|---|
| 2026-11-15 | Claude (Anthropic), Claude Opus 5.5 | Відновлення формату архіву розбором еталонних архівів викладача (ключ з паролю в team_config.json, розшифрування, аналіз структури payload) | `docs/FORMAT_SPEC.md` |
| 2026-11-15 | Claude (Anthropic), Claude Opus 5.5 | Виправлення за рев'ю PR #1 | `sarc/*.py`, `tests/*.py` |
| 2026-11-16 | Claude (Anthropic), Claude Opus 5.5 | Виправлення за рев'ю PR #2: ДП за цінністю, тимчасовий файл для блоків, обробка помилок читання/запису, медіана часу | `sarc/packing.py`, `sarc/archive.py`, `sarc/lz77.py`, `tools/measure.py`, `tests/*.py` |
| 2026-11-16 | Claude (Anthropic), Claude Opus 5.5 | Архіватор: кодер Гаффмана, LZ77 (жадібний і лінивий), рюкзак ДП/жадібний, команда archive, тести 12 кейсів C1, скрипт вимірювань | `sarc/huffman.py`, `sarc/codec.py`, `sarc/lz77.py`, `sarc/packing.py`, `sarc/analysis.py`, `sarc/archive.py`, `sarc/cli.py`, `tests/test_acceptance.py`, `tools/measure.py`, `docs/RESULTS.md` |
| 2026-11-15 | Claude (Anthropic), Claude Opus 5.5 | Каркас пакета, розархіватор з кодами відмов, тести на еталонних архівах, CI | `sarc/*.py`, `tests/*.py`, `.github/workflows/ci.yml`, `pyproject.toml`, `README.md` |
| 2026-11-16 | Claude (Anthropic), Claude Opus 5.5 | Звірка репозиторію з методичкою (розділи 3, 6, 10) і виправлення невідповідностей: суворий декодер (коди ≤ 15 біт, рівність Крафта, алфавіти, вікно 32768), код E_CODE, порожній архів, межі LZParams, нові тести, лог рішень, перенесення `tools/` → `experiments/` | `sarc/huffman.py`, `sarc/codec.py`, `sarc/errors.py`, `sarc/lz77.py`, `sarc/cli.py`, `sarc/archive.py`, `sarc/packing.py`, `tests/*.py`, `docs/DECISIONS.md`, `docs/FORMAT_SPEC.md`, `README.md` |
| 2026-11-16 | Claude (Anthropic), Claude Opus 5.5 | Класифікатор типу файлу: ознаки з байтів, перцептрон (forward, softmax, cross-entropy, backprop, SGD) на numpy, macro-F1 і confusion matrix, навчання з 3 ініціалізаціями, підключення до архіватора, вимір «клас vs фіксована стратегія», тести (зокрема числова перевірка градієнта) | `sarc/features.py`, `sarc/mlp.py`, `sarc/mlp_model.json`, `sarc/analysis.py`, `sarc/archive.py`, `experiments/train_mlp.py`, `experiments/measure.py`, `tests/test_mlp.py`, `docs/MLP_RESULTS.md`, `docs/RESULTS.md`, `docs/DECISIONS.md` |
| 2026-11-17 | Claude (Anthropic), Claude Opus 5.5 | Виправлення за рев'ю #5: код E_MODEL для відсутньої/битої моделі, зразок для класифікатора незалежно від block_size (_blocks), розділ «Стійкість» у звіті, тести | `sarc/analysis.py`, `sarc/archive.py`, `sarc/errors.py`, `experiments/train_mlp.py`, `tests/test_mlp.py`, `docs/MLP_RESULTS.md`, `docs/DECISIONS.md`, `README.md` |
| 2026-11-17 | Claude (Anthropic), Claude Opus 5.5 | Код відмови E_VOLUMES за методичкою (10.53) замість E_VOLUME; volume_index ≥ volume_count -> E_VOLUMES; тести | `sarc/errors.py`, `sarc/extract.py`, `sarc/container.py`, `tests/*.py`, `README.md`, `docs/FORMAT_SPEC.md`, `docs/DECISIONS.md` |
| 2026-11-17 | Claude (Anthropic), Claude Opus 5.5 | Генетичний алгоритм і random search (селекція, схрещування, мутація, елітизм, бюджет і кеш оцінок), експеримент на корпусі, почерговий перевимір переможця, нові типові параметри LZ77, тести ГА за методичкою (6.14) | `sarc/ga.py`, `sarc/lz77.py`, `sarc/cli.py`, `experiments/run_ga.py`, `experiments/measure.py`, `tests/test_ga.py`, `docs/GA_RESULTS.md`, `docs/RESULTS.md`, `docs/DECISIONS.md`, `README.md` |
| 2026-11-17 | Claude (Anthropic), Claude Opus 5.5 | Прискорення пошуку збігів LZ77 (профілювання, вбудовування хешу і вставки в longest), тест проти повного перебору | `sarc/lz77.py`, `tests/test_primitives.py`, `docs/DECISIONS.md` |
| 2026-11-17 | Claude (Anthropic), Claude Opus 5.5 | Модуль атаки: навчальний XOR-шифр, модель тексту, пошук ключа ГА (мутація reset у sarc/ga.py), контроль на справжньому архіві з PIN, випадкові пошкодження архіву, експерименти й тести | `attack/*.py`, `sarc/ga.py`, `experiments/run_attack.py`, `tests/test_attack.py`, `docs/ATTACK_RESULTS.md`, `docs/DECISIONS.md`, `README.md` |
| 2026-11-17 | Claude (Anthropic), Claude Opus 5.5 | Вимір залежності типу файлу від позиції зразка і рішення лишити перші 16 КБ | `experiments/sample_position.py`, `docs/DECISIONS.md` |
| 2026-11-17 | Claude (Anthropic), Claude Opus 5.5 | Чернетка звіту за розділом 8 методички; перегенерація результатів на фінальному main | `docs/REPORT.md`, `docs/RESULTS.md`, `docs/GA_RESULTS.md`, `experiments/results/ga_runs.json` |

Перевірку виконано автоматичними тестами: усі еталонні архіви `data/reference/ref_*.arc`
розпаковуються з правильними CRC32, усі `bad_*.arc` дають очікувані коди з `expected_failures.json`.

Кожен учасник, що використовує ШІ у своїй зоні, додає сюди рядок: дата, інструмент, що саме, які файли.
