"""Графічний інтерфейс архіватора (tkinter).

    python -m sarc gui        або        python -m sarc.gui

GUI — лише оболонка: усю роботу роблять ті самі create_archive і extract, що й у командного рядка.
Тому GUI не може зіпсувати архів інакше, ніж CLI, а помилки показує тими самими кодами (E_HMAC тощо).

Будова:
  - archive_action / extract_action — звичайні функції без tkinter: перевіряють введене, викликають
    архіватор і повертають текст результату. Їх перевіряють тести без дисплея.
  - App — вікно з двома вкладками. Довга операція йде у фоновому потоці, щоб вікно не «зависало»;
    результат потік кладе в чергу, а вікно раз на 100 мс перевіряє чергу (tkinter не можна чіпати з
    іншого потоку — лише з головного).
"""
import queue
import threading

from .archive import Options, create_archive
from .cli import resolve_inputs, resolve_volumes
from .errors import E_INPUT, ArchiveError
from .extract import extract

DEFAULT_VOLUME = Options().max_volume


# ---------- дії (без tkinter) ----------

def archive_action(inputs: list[str], out_path: str, password: str, confirm: str,
                   volume_size: str, overwrite: bool) -> str:
    """Перевірити введене, створити архів. Повертає текст для вікна; помилка -> ArchiveError."""
    if not inputs:
        raise ArchiveError(E_INPUT, "не вибрано жодного файлу чи папки")
    if not out_path:
        raise ArchiveError(E_INPUT, "не вказано, куди зберегти архів")
    if not password:
        raise ArchiveError(E_INPUT, "пароль порожній")
    if password != confirm:
        raise ArchiveError(E_INPUT, "паролі не збігаються")
    try:
        size = int(volume_size)
    except ValueError:
        raise ArchiveError(E_INPUT, f"розмір тому має бути цілим числом, а не «{volume_size}»") from None
    res = create_archive(resolve_inputs(inputs), out_path, password, Options(max_volume=size),
                         overwrite=overwrite)
    ratio = res.original_bytes / res.archive_bytes if res.archive_bytes else 0
    lines = [f"Готово: {len(res.files)} файлів, {res.original_bytes} → {res.archive_bytes} байт, "
             f"коефіцієнт {ratio:.3f}", f"Томів: {len(res.volumes)}"]
    return "\n".join(lines + [f"  {v}" for v in res.volumes])


def extract_action(volumes: list[str], password: str, out_dir: str, overwrite: bool) -> str:
    if not volumes:
        raise ArchiveError(E_INPUT, "не вибрано жодного тому")
    if not out_dir:
        raise ArchiveError(E_INPUT, "не вказано папку для файлів")
    names = extract(resolve_volumes(volumes), password, out_dir, overwrite=overwrite)
    return "\n".join([f"Готово: відновлено {len(names)} файлів у {out_dir}"] + [f"  {n}" for n in names])


def run_safely(action, *args) -> tuple[bool, str]:
    """(успіх, текст). Перехоплює контрольовані помилки й помилки файлової системи."""
    try:
        return True, action(*args)
    except ArchiveError as e:
        return False, str(e)
    except OSError as e:
        return False, f"{E_INPUT}: {e.strerror}: {e.filename}"


# ---------- вікно ----------

def main() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk
    except ImportError:
        print("tkinter не встановлено: на Windows і macOS він входить у Python з python.org, "
              "на Linux — пакет python3-tk.")
        return 2

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("SARC — розумний захищений архіватор")
            self.minsize(620, 440)
            self.results: queue.Queue = queue.Queue()
            self.busy = False
            tabs = ttk.Notebook(self)
            tabs.pack(fill="both", expand=True, padx=8, pady=8)
            tabs.add(self._archive_tab(tabs), text="Архівувати")
            tabs.add(self._extract_tab(tabs), text="Розпакувати")
            self.status = tk.Text(self, height=8, state="disabled", wrap="word")
            self.status.pack(fill="both", padx=8, pady=(0, 8))
            self.after(100, self._poll)

        # --- допоміжні елементи ---

        def _row(self, parent, r, label, var, browse=None, show=None):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 10))
            entry = ttk.Entry(parent, textvariable=var, width=55, show=show)
            entry.grid(row=r, column=1, sticky="we", pady=3)
            if browse:
                ttk.Button(parent, text="…", width=3, command=browse).grid(row=r, column=2, padx=4)

        def _show(self, ok: bool, text: str):
            self.status.configure(state="normal")
            self.status.delete("1.0", "end")
            self.status.insert("end", ("✅ " if ok else "❌ ") + text)
            self.status.configure(state="disabled")

        # --- вкладка «Архівувати» ---

        def _archive_tab(self, parent):
            f = ttk.Frame(parent, padding=10)
            f.columnconfigure(1, weight=1)
            self.a_inputs: list[str] = []
            self.a_inputs_text = tk.StringVar(value="(нічого не вибрано)")
            self.a_out, self.a_pw, self.a_pw2 = tk.StringVar(), tk.StringVar(), tk.StringVar()
            self.a_size, self.a_force = tk.StringVar(value=str(DEFAULT_VOLUME)), tk.BooleanVar()
            ttk.Label(f, text="Що архівувати").grid(row=0, column=0, sticky="w", padx=(0, 10))
            ttk.Label(f, textvariable=self.a_inputs_text).grid(row=0, column=1, sticky="w")
            btns = ttk.Frame(f)
            btns.grid(row=1, column=1, sticky="w", pady=3)
            ttk.Button(btns, text="Додати файли", command=self._add_files).pack(side="left")
            ttk.Button(btns, text="Додати папку", command=self._add_folder).pack(side="left", padx=4)
            ttk.Button(btns, text="Очистити", command=self._clear_inputs).pack(side="left")
            self._row(f, 2, "Зберегти як", self.a_out, self._choose_out)
            self._row(f, 3, "Пароль", self.a_pw, show="•")
            self._row(f, 4, "Пароль ще раз", self.a_pw2, show="•")
            self._row(f, 5, "Розмір тому, байт", self.a_size)
            ttk.Checkbutton(f, text="Перезаписати наявні томи", variable=self.a_force).grid(
                row=6, column=1, sticky="w")
            go = ttk.Button(f, text="Архівувати", command=self._do_archive)
            go.grid(row=7, column=1, sticky="e", pady=8)
            return f

        def _add_files(self):
            self.a_inputs += [p for p in filedialog.askopenfilenames(title="Файли") if p not in self.a_inputs]
            self._update_inputs()

        def _add_folder(self):
            d = filedialog.askdirectory(title="Папка")
            if d and d not in self.a_inputs:
                self.a_inputs.append(d)
            self._update_inputs()

        def _clear_inputs(self):
            self.a_inputs = []
            self._update_inputs()

        def _update_inputs(self):
            n = len(self.a_inputs)
            self.a_inputs_text.set("(нічого не вибрано)" if not n else
                                   self.a_inputs[0] if n == 1 else f"{self.a_inputs[0]} і ще {n - 1}")

        def _choose_out(self):
            p = filedialog.asksaveasfilename(title="Зберегти архів", defaultextension=".arc",
                                             filetypes=[("Архів SARC", "*.arc")])
            if p:
                self.a_out.set(p)

        def _do_archive(self):
            self._start(archive_action, list(self.a_inputs), self.a_out.get(), self.a_pw.get(),
                        self.a_pw2.get(), self.a_size.get(), self.a_force.get())

        # --- вкладка «Розпакувати» ---

        def _extract_tab(self, parent):
            f = ttk.Frame(parent, padding=10)
            f.columnconfigure(1, weight=1)
            self.e_volumes: list[str] = []
            self.e_vol_text = tk.StringVar(value="(нічого не вибрано)")
            self.e_pw, self.e_out, self.e_force = tk.StringVar(), tk.StringVar(), tk.BooleanVar()
            ttk.Label(f, text="Томи архіву").grid(row=0, column=0, sticky="w", padx=(0, 10))
            ttk.Label(f, textvariable=self.e_vol_text).grid(row=0, column=1, sticky="w")
            ttk.Button(f, text="Вибрати томи", command=self._choose_volumes).grid(
                row=1, column=1, sticky="w", pady=3)
            self._row(f, 2, "Пароль", self.e_pw, show="•")
            self._row(f, 3, "Папка для файлів", self.e_out, self._choose_dir)
            ttk.Checkbutton(f, text="Перезаписати наявні файли", variable=self.e_force).grid(
                row=4, column=1, sticky="w")
            go = ttk.Button(f, text="Розпакувати", command=self._do_extract)
            go.grid(row=5, column=1, sticky="e", pady=8)
            return f

        def _choose_volumes(self):
            chosen = filedialog.askopenfilenames(title="Томи", filetypes=[("Архів SARC", "*.arc")])
            self.e_volumes = list(chosen)
            n = len(self.e_volumes)
            self.e_vol_text.set("(нічого не вибрано)" if not n else f"{n} том(и): {self.e_volumes[0]}"
                                + (" …" if n > 1 else ""))

        def _choose_dir(self):
            d = filedialog.askdirectory(title="Папка для файлів")
            if d:
                self.e_out.set(d)

        def _do_extract(self):
            self._start(extract_action, list(self.e_volumes), self.e_pw.get(), self.e_out.get(),
                        self.e_force.get())

        # --- фоновий потік ---

        def _start(self, action, *args):
            if self.busy:
                return
            self.busy = True
            self.configure(cursor="watch")
            self._show(True, "Працюю…")
            threading.Thread(target=lambda: self.results.put(run_safely(action, *args)), daemon=True).start()

        def _poll(self):
            try:
                ok, text = self.results.get_nowait()
            except queue.Empty:
                pass
            else:
                self.busy = False
                self.configure(cursor="")
                self._show(ok, text)
            self.after(100, self._poll)

    App().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
