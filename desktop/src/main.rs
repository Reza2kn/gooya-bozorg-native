use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

use eframe::egui::{
    self, Align, Color32, FontData, FontDefinitions, FontFamily, FontId, Frame, Layout,
    text::{LayoutJob, TextFormat},
    Margin, Rect, RichText, Stroke, TextEdit, UiBuilder, Vec2,
};

const BURGUNDY: Color32 = Color32::from_rgb(99, 14, 28);
const INK: Color32 = Color32::from_rgb(31, 24, 20);
const PAPER: Color32 = Color32::from_rgb(246, 241, 231);

fn display_rtl(text: &str) -> String {
    arabic_reshaper::arabic_reshape(text)
        .split('\n')
        .map(|line| {
            line.split_whitespace()
                .rev()
                .map(|word| word.chars().rev().collect::<String>())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn rtl_layouter(ui: &egui::Ui, text: &str, _wrap_width: f32) -> Arc<egui::Galley> {
    let mut job = LayoutJob::single_section(
        display_rtl(text),
        TextFormat::simple(FontId::proportional(21.0), INK),
    );
    job.wrap.max_width = f32::INFINITY;
    ui.fonts(|fonts| fonts.layout_job(job))
}

enum JobMessage {
    Done(Result<gooya_native_desktop::pipeline::SynthesisReport, String>),
}

struct GooyaApp {
    text: String,
    status: String,
    busy: bool,
    model_dir: PathBuf,
    tokenizer_path: PathBuf,
    last_wav: Option<PathBuf>,
    receiver: Option<mpsc::Receiver<JobMessage>>,
}

impl Default for GooyaApp {
    fn default() -> Self {
        let root = data_root();
        Self {
            text: String::new(),
            status: "متن را وارد کنید".to_owned(),
            busy: false,
            model_dir: root.join("tract-bundle-b168"),
            tokenizer_path: root.join("grapheme_mtl_merged_expanded_v1.json"),
            last_wav: None,
            receiver: None,
        }
    }
}

/// Resolve the model data directory: $GOOYA_MODEL_DIR, else a `data/` folder
/// shipped next to the executable, else the crate's checked-in `data/`.
fn data_root() -> PathBuf {
    if let Ok(dir) = std::env::var("GOOYA_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    let exe = std::env::current_exe().unwrap_or_default();
    let mut dir = exe.parent().unwrap_or(&exe).to_path_buf();
    if dir.ends_with("release") || dir.ends_with("debug") {
        dir = dir.join("data");
    }
    let candidate = dir.join("tract-bundle-b168");
    if candidate.is_dir() {
        return dir;
    }
    // Dev fallback: checked-in data next to Cargo.toml.
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dev = crate_dir.join("data");
    if dev.join("tract-bundle-b168").is_dir() {
        return dev;
    }
    crate_dir
}

impl GooyaApp {
    fn start_job(&mut self) {
        let model_dir = self.model_dir.clone();
        let tokenizer_path = self.tokenizer_path.clone();
        let text = self.text.trim().to_owned();
        let out = std::env::temp_dir().join("gooya-desktop-output.wav");
        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.busy = true;
        self.status = "در حال ساخت صدا…".to_owned();
        self.last_wav = None;
        std::thread::spawn(move || {
            let result = (|| -> Result<gooya_native_desktop::pipeline::SynthesisReport, String> {
                gooya_native_desktop::pipeline::synthesize_text(&model_dir, &tokenizer_path, &out, &text)
                    .map_err(|e| format!("synthesize: {e:#}"))
            })();
            let _ = tx.send(JobMessage::Done(result));
        });
    }

    fn poll(&mut self) {
        if let Some(rx) = &self.receiver {
            if let Ok(JobMessage::Done(result)) = rx.try_recv() {
                self.receiver = None;
                self.busy = false;
                match result {
                    Ok(report) => {
                        self.status = format!(
                            "خوانده شد · {:.2} ثانیه · peak {:.2} · {}",
                            report.duration_seconds, report.peak, report.wav_path.display()
                        );
                        self.last_wav = Some(report.wav_path.clone());
                        play(&report.wav_path);
                    }
                    Err(err) => {
                        self.status = format!("خطا: {err}");
                    }
                }
            }
        }
    }

    fn draw_fixed(&mut self, context: &egui::Context) {
        egui::CentralPanel::default()
            .frame(Frame::new().fill(PAPER).inner_margin(Margin::same(28)))
            .show(context, |ui| {
                let available = ui.available_rect_before_wrap();
                let width = available.width().min(760.0);
                let editor_height = (available.height() - 360.0).clamp(120.0, 160.0);
                let card_height = editor_height + 74.0;
                let total_height = 18.0 + 60.0 + 8.0 + 24.0 + 16.0 + card_height + 10.0 + 24.0 + 10.0 + 60.0 + 40.0 + 18.0;
                let left = available.center().x - width / 2.0;
                let top = available.center().y - total_height / 2.0;
                let row = |y: f32, height: f32| {
                    Rect::from_min_size(egui::pos2(left, y), Vec2::new(width, height))
                };

                ui.allocate_ui_at_rect(row(top + 18.0, 60.0), |ui| {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(RichText::new(display_rtl("گویا")).font(FontId::proportional(54.0)).strong().color(INK));
                        ui.add_space(16.0);
                        ui.label(RichText::new("BOZORG · 1.5").font(FontId::monospace(11.0)).color(BURGUNDY.gamma_multiply(0.72)));
                    });
                });
                ui.allocate_ui_at_rect(row(top + 86.0, 24.0), |ui| {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(RichText::new(display_rtl("خوانش آفلاین فارسی، روی همین دستگاه")).size(15.0).color(INK.gamma_multiply(0.58)));
                    });
                });
                let card_top = top + 126.0;
                let card_rect = row(card_top, card_height);
                ui.painter().rect_filled(
                    card_rect,
                    20.0,
                    Color32::from_rgba_unmultiplied(255, 255, 255, 235),
                );
                ui.painter().rect_stroke(
                    card_rect,
                    20.0,
                    Stroke::new(1.0, BURGUNDY.gamma_multiply(0.18)),
                    egui::StrokeKind::Outside,
                );
                let inner = 18.0;
                let label_row = Rect::from_min_size(
                    egui::pos2(card_rect.left() + inner, card_rect.top() + inner),
                    Vec2::new(card_rect.width() - 2.0 * inner, 20.0),
                );
                let editor_rect = Rect::from_min_size(
                    egui::pos2(card_rect.left() + inner, label_row.bottom() + 8.0),
                    Vec2::new(card_rect.width() - 2.0 * inner, card_rect.bottom() - inner - 8.0 - label_row.height()),
                );
                ui.allocate_ui_at_rect(label_row, |ui| {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(RichText::new(display_rtl("متن ورودی")).size(13.0).strong().color(BURGUNDY));
                        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                            ui.label(RichText::new(display_rtl(&format!("{} نویسه", self.text.chars().count()))).size(11.0).color(INK.gamma_multiply(0.42)));
                        });
                    });
                });
                let editor_size = editor_rect.size();
                ui.allocate_ui_at_rect(editor_rect, |ui| {
                    ui.add_sized(
                        editor_size,
                        TextEdit::multiline(&mut self.text)
                            .hint_text(display_rtl("مثلاً: امروز هوا چقدر دل‌انگیز است…"))
                            .font(FontId::proportional(21.0))
                            .horizontal_align(Align::RIGHT)
                            .vertical_align(Align::Min)
                            .desired_rows(0)
                            .desired_width(0.0)
                            .margin(Margin::ZERO)
                            .interactive(!self.busy)
                            .frame(false),
                    );
                });
                let status_top = card_top + card_height + 10.0;
                ui.allocate_ui_at_rect(row(status_top, 24.0), |ui| {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if self.busy { ui.spinner(); ui.add_space(8.0); }
                        ui.label(RichText::new(display_rtl(&self.status)).size(13.0).color(INK.gamma_multiply(0.52)));
                    });
                });
                let button_top = status_top + 34.0;
                let enabled = !self.busy && !self.text.trim().is_empty();
                let button = egui::Button::new(RichText::new(display_rtl("بگو")).font(FontId::proportional(28.0)).strong().color(Color32::WHITE))
                    .fill(if self.busy { BURGUNDY.gamma_multiply(0.62) } else if enabled { BURGUNDY } else { BURGUNDY.gamma_multiply(0.42) })
                    .corner_radius(18.0)
                    .min_size(Vec2::new(width, 60.0));
                if ui.put(row(button_top, 60.0), button).clicked() && enabled { self.start_job(); }
                if let Some(wav) = &self.last_wav {
                    if ui.put(row(button_top + 68.0, 28.0), egui::Button::new(RichText::new(display_rtl("دوباره پخش کن")).color(INK.gamma_multiply(0.6)))).clicked() { play(wav); }
                }
                ui.allocate_ui_at_rect(row(top + total_height - 18.0, 18.0), |ui| {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(RichText::new(display_rtl("اجرای محلی · بدون ارسال متن به اینترنت")).size(11.0).color(INK.gamma_multiply(0.38)));
                    });
                });
            });
    }
}

fn play(path: &std::path::Path) {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = std::process::Command::new("afplay");
        command.arg(path);
        command
    };
    #[cfg(target_os = "linux")]
    let mut command = {
        let mut command = std::process::Command::new("paplay");
        command.arg(path);
        command
    };
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = std::process::Command::new("powershell");
        command.args([
            "-NoProfile",
            "-Command",
            &format!(
                "(New-Object Media.SoundPlayer '{}').PlaySync()",
                path.display()
            ),
        ]);
        command
    };
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = path;
        return;
    }
    let _ = command.status();
}

impl eframe::App for GooyaApp {
    #[allow(unreachable_code)]
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        self.draw_fixed(context);
        return;
        egui::CentralPanel::default()
            .frame(Frame::new().fill(PAPER).inner_margin(Margin::same(28)))
            .show(context, |ui| {
                let available = ui.available_rect_before_wrap();
                let content_size = Vec2::new(
                    available.width().min(760.0),
                    available.height().min(560.0),
                );
                let content_rect = Rect::from_center_size(available.center(), content_size);
                ui.scope_builder(
                        UiBuilder::new()
                        .max_rect(content_rect)
                        .layout(Layout::top_down(Align::Center).with_main_align(Align::Min)),
                    |ui| {
                            ui.add_space(18.0);
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.label(
                                    RichText::new(display_rtl("گویا"))
                                        .font(FontId::proportional(54.0))
                                        .strong()
                                        .color(INK),
                                );
                                ui.add_space(16.0);
                                ui.label(
                                    RichText::new("BOZORG · 1.5")
                                        .font(FontId::monospace(11.0))
                                        .color(BURGUNDY.gamma_multiply(0.72)),
                                );
                            });
                            ui.add_space(4.0);
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.label(
                                    RichText::new(display_rtl("خوانش آفلاین فارسی، روی همین دستگاه"))
                                        .size(15.0)
                                        .color(INK.gamma_multiply(0.58)),
                                );
                            });
                            ui.add_space(24.0);

                            Frame::new()
                                .fill(Color32::from_rgba_unmultiplied(255, 255, 255, 235))
                                .stroke(Stroke::new(1.0, BURGUNDY.gamma_multiply(0.18)))
                                .corner_radius(20.0)
                                .inner_margin(Margin::same(18))
                                .show(ui, |ui| {
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        ui.label(
                                            RichText::new(display_rtl("متن ورودی"))
                                                .size(13.0)
                                                .strong()
                                                .color(BURGUNDY),
                                        );
                                        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                                            ui.label(
                                                RichText::new(display_rtl(&format!(
                                                    "{} نویسه",
                                                    self.text.chars().count()
                                                )))
                                                    .size(11.0)
                                                    .color(INK.gamma_multiply(0.42)),
                                            );
                                        });
                                    });
                                    ui.add_space(12.0);
                                    ui.add_sized(
                                        Vec2::new(ui.available_width(), 160.0),
                                        TextEdit::multiline(&mut self.text)
                                            .hint_text(display_rtl("مثلاً: امروز هوا چقدر دل‌انگیز است…"))
                                            .font(FontId::proportional(21.0))
                                            .horizontal_align(Align::RIGHT)
                                            .layouter(&mut |ui, text, wrap_width| {
                                                rtl_layouter(ui, text, wrap_width)
                                            })
                                            .interactive(!self.busy)
                                            .frame(false),
                                    );
                                });

                            ui.add_space(12.0);
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if self.busy {
                                    ui.spinner();
                                    ui.add_space(8.0);
                                }
                                ui.label(
                                    RichText::new(display_rtl(&self.status))
                                        .size(13.0)
                                        .color(INK.gamma_multiply(0.52)),
                                );
                            });
                            ui.add_space(12.0);

                            let enabled = !self.busy && !self.text.trim().is_empty();
                            let button = egui::Button::new(
                                RichText::new(display_rtl("بگو"))
                                    .font(FontId::proportional(28.0))
                                    .strong()
                                    .color(Color32::WHITE),
                            )
                            .fill(if self.busy {
                                BURGUNDY.gamma_multiply(0.62)
                            } else if enabled {
                                BURGUNDY
                            } else {
                                BURGUNDY.gamma_multiply(0.42)
                            })
                            .corner_radius(18.0)
                            .min_size(Vec2::new(ui.available_width(), 60.0));
                            if ui.add_enabled(enabled, button).clicked() {
                                self.start_job();
                            }

                            if let Some(wav) = &self.last_wav {
                                ui.add_space(8.0);
                                if ui
                                    .button(RichText::new(display_rtl("دوباره پخش کن")).color(INK.gamma_multiply(0.6)))
                                    .clicked()
                                {
                                    play(wav);
                                }
                            }

                            ui.add_space(28.0);
                            ui.label(
                                RichText::new("اجرای محلی · بدون ارسال متن به اینترنت")
                                    .size(11.0)
                                    .color(INK.gamma_multiply(0.38)),
                            );
                });
            });
    }
}

fn install_fonts(context: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "vazirmatn".to_owned(),
        Arc::new(FontData::from_static(include_bytes!(
            "../../../GooyaCoreMLDemo/Gooya/Resources/Fonts/Vazirmatn-Regular.ttf"
        ))),
    );
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        fonts.families.entry(family).or_default().insert(0, "vazirmatn".to_owned());
    }
    context.set_fonts(fonts);
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([620.0, 720.0])
            .with_min_inner_size([440.0, 600.0])
            .with_title("گویا"),
        ..Default::default()
    };
    eframe::run_native(
        "گویا",
        options,
        Box::new(|creation| {
            install_fonts(&creation.egui_ctx);
            Ok(Box::<GooyaApp>::default())
        }),
    )
}
