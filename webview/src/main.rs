use std::path::PathBuf;
use std::process::Command;

use anyhow::Result;
use tao::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy, EventLoopWindowTarget},
    window::{Window, WindowBuilder},
};
use wry::WebViewBuilder;

use gooya_native_desktop::pipeline;

enum UserEvent {
    Done(Result<String, String>),
    Paste(Result<String, String>),
}

fn main() -> wry::Result<()> {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();

    let (_window, webview) = create_window(&event_loop, proxy.clone());

    event_loop.run(move |event, _event_loop, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::UserEvent(UserEvent::Done(status)) => {
                let (text, ok) = match status {
                    Ok(text) => (text, true),
                    Err(text) => (text, false),
                };
                let script = format!(
                    "window.__gooyaResult({}, {})",
                    if ok { "true" } else { "false" },
                    serde_json::to_string(&text).unwrap_or_else(|_| "\"خطا\"".into())
                );
                let _ = webview.evaluate_script(&script);
            }
            Event::UserEvent(UserEvent::Paste(result)) => {
                let text = result.unwrap_or_else(|_| String::new());
                let script = format!(
                    "window.__gooyaPaste({})",
                    serde_json::to_string(&text).unwrap_or_else(|_| "\"\"".into())
                );
                let _ = webview.evaluate_script(&script);
            }
            _ => {}
        }
    });
}

fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("GOOYA_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    // Packaged app bundle: <Gooya.app>/Contents/data
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let bundled = parent.join("../data");
            if bundled.join("tract-bundle-b168").is_dir() {
                return bundled;
            }
        }
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = crate_dir.join("../desktop/data");
    if candidate.join("tract-bundle-b168").is_dir() {
        return candidate;
    }
    candidate
}

fn ensure_wav_exists(path: &std::path::Path) -> Result<()> {
    anyhow::ensure!(path.exists(), "no audio has been generated yet");
    Ok(())
}

fn play(path: std::path::PathBuf) {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut c = Command::new("afplay");
        c.arg(&path);
        c
    };
    #[cfg(target_os = "linux")]
    let mut command = {
        let mut c = Command::new("paplay");
        c.arg(&path);
        c
    };
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut c = Command::new("powershell");
        c.args([
            "-NoProfile",
            "-Command",
            &format!("(New-Object Media.SoundPlayer '{}').PlaySync()", path.display()),
        ]);
        c
    };
    let _ = command.status();
}

fn create_window(
    event_loop: &EventLoopWindowTarget<UserEvent>,
    proxy: EventLoopProxy<UserEvent>,
) -> (Window, wry::WebView) {
    let window = WindowBuilder::new()
        .with_title("گویا")
        .with_inner_size(tao::dpi::LogicalSize::new(660.0, 760.0))
        .with_min_inner_size(tao::dpi::LogicalSize::new(460.0, 620.0))
        .build(event_loop)
        .unwrap();

    let root = data_dir();
    let model_dir = root.join("tract-bundle-b168");
    let tokenizer_path = root.join("grapheme_mtl_merged_expanded_v1.json");
    let model_dir_for_job = model_dir.clone();
    let tokenizer_for_job = tokenizer_path.clone();

    let handler = move |req: wry::http::Request<String>| {
        let body = req.body().clone();
        let proxy = proxy.clone();
        if body == "paste" {
            let result = arboard::Clipboard::new()
                .and_then(|mut clipboard| clipboard.get_text())
                .map_err(|error| format!("{error}"));
            let _ = proxy.send_event(UserEvent::Paste(result));
            return;
        }
        if body == "replay" {
            let wav = std::env::temp_dir().join("gooya-webview-output.wav");
            if wav.exists() {
                play(wav);
            }
            return;
        }
        if body == "save" {
            let wav = std::env::temp_dir().join("gooya-webview-output.wav");
            std::thread::spawn(move || {
                let result = (|| -> Result<String> {
                    ensure_wav_exists(&wav)?;
                    let file = rfd::FileDialog::new()
                        .add_filter("WAV", &["wav"])
                        .set_file_name("gooya.wav")
                        .save_file();
                    let Some(file) = file else {
                        return Ok(String::from("ذخیره نشد"));
                    };
                    std::fs::copy(&wav, &file)?;
                    let name = file
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| file.display().to_string());
                    Ok(format!("ذخیره شد: {name}"))
                })();
                let _ = proxy.send_event(UserEvent::Done(result.map_err(|e| format!("{e:#}"))));
            });
            return;
        }
        if !body.starts_with("text:") {
            return;
        }
        let text = body.trim_start_matches("text:").to_owned();
        let model_dir = model_dir_for_job.clone();
        let tokenizer_path = tokenizer_for_job.clone();
        std::thread::spawn(move || {
            let out = std::env::temp_dir().join("gooya-webview-output.wav");
            let result = (|| -> Result<String> {
                let report = pipeline::synthesize_text(&model_dir, &tokenizer_path, &out, &text)?;
                play(out.clone());
                Ok(format!(
                    "{:.2}s · {}",
                    report.duration_seconds,
                    report.wav_path.display()
                ))
            })();
            let _ = proxy.send_event(UserEvent::Done(result.map_err(|e| format!("{e:#}"))));
        });
    };

    let builder = WebViewBuilder::new()
        .with_html(app_html())
        .with_ipc_handler(handler);

    #[cfg(any(target_os = "windows", target_os = "macos"))]
    let webview = builder.build(&window).unwrap();
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "ios", target_os = "android")))]
    let webview = {
        use tao::platform::unix::WindowExtUnix;
        use wry::WebViewBuilderExtUnix;
        let vbox = window.default_vbox().unwrap();
        builder.build_gtk(vbox).unwrap()
    };
    (window, webview)
}

const FONT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../GooyaCoreMLDemo/Gooya/Resources/Fonts/Vazirmatn-Regular.ttf"
);

fn app_html() -> String {
    use base64::Engine as _;
    let font_bytes = include_bytes!("../../../GooyaCoreMLDemo/Gooya/Resources/Fonts/Vazirmatn-Regular.ttf");
    let _ = FONT_PATH;
    let font_uri = format!(
        "data:font/ttf;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(font_bytes)
    );
    HTML.replace("__FONT_DATA_URI__", &font_uri)
}

const HTML: &str = r#"
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>گویا</title>
<style>
  @font-face{font-family:"Vazirmatn";src:url(__FONT_DATA_URI__) format("truetype");font-weight:100 900;}
  :root{--burgundy:#631e1c;--ink:#1f1814;--paper:#f6f1e7;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--paper);color:var(--ink);
       font-family:"Vazirmatn",-apple-system,"Segoe UI",Tahoma,sans-serif;
       height:100vh;display:flex;align-items:center;justify-content:center;
       padding:2vh 4vw;}
  .page{width:min(100%,740px);}
  .word{font-size:64px;font-weight:800;display:inline;vertical-align:middle;line-height:1.1;}
  .tag{font-family:monospace;font-size:13px;color:var(--burgundy);opacity:.72;margin-right:16px;}
  .sub{font-size:17px;color:var(--ink);opacity:.58;margin:10px 0 26px;}
  .card{background:rgba(255,255,255,.92);border:1px solid rgba(99,30,28,.18);
        border-radius:22px;padding:22px;text-align:right;}
  .cardhead{display:flex;justify-content:space-between;align-items:center;
            color:var(--burgundy);font-size:14px;font-weight:700;}
  .count{color:var(--ink);opacity:.42;font-size:12px;}
  textarea{width:100%;min-height:220px;margin-top:14px;border:0;outline:0;resize:none;
           font-size:24px;line-height:2;font-family:inherit;color:var(--ink);}
  textarea::placeholder{color:var(--ink);opacity:.35;}
  .status{margin:16px 0;min-height:22px;font-size:14px;color:var(--ink);opacity:.52;
          display:flex;align-items:center;justify-content:center;gap:8px;}
  .spin{width:18px;height:18px;border:2px solid rgba(99,30,28,.3);border-top-color:var(--burgundy);
        border-radius:50%;animation:sp 1s linear infinite;}
  @keyframes sp{to{transform:rotate(360deg)}}
  button{width:100%;border:0;border-radius:18px;padding:20px;font-size:28px;font-weight:800;
         color:#fff;background:var(--burgundy);cursor:pointer;font-family:inherit;}
  button:disabled{opacity:.45;cursor:default;}
  .row{display:flex;gap:10px;}
  .ghost{flex:1;margin-top:10px;background:transparent;color:var(--ink);opacity:.75;
         font-size:15px;font-weight:700;border:1px solid rgba(31,24,20,.2);}
  .foot{margin-top:24px;font-size:12px;color:var(--ink);opacity:.38;text-align:center;}
</style>
</head>
<body>
<div class="page">
  <div><span class="word">گویا</span><span class="tag">BOZORG · 1.5</span></div>
  <div class="sub">خوانش آفلاین فارسی، روی همین دستگاه</div>
  <div class="card">
    <div class="cardhead"><span>متن ورودی</span><span class="count" id="count">۰ نویسه</span></div>
    <textarea id="t" dir="rtl" autofocus
      placeholder="مثلاً: امروز هوا چقدر دل‌انگیز است…"
      oninput="var c=enDigits(document.getElementById('t').value.length);document.getElementById('count').textContent=c+' نویسه';">سلام، حالت چطوره؟</textarea>
  </div>
  <div class="status" id="status">متن را وارد کنید</div>
  <button id="go" dir="rtl" onclick="speak()">بگو</button>
  <div class="row" id="postrow">
    <button class="ghost" id="replay" dir="rtl" onclick="window.ipc.postMessage('replay')">دوباره پخش کن</button>
    <button class="ghost" id="save" dir="rtl" onclick="saveIt()">ذخیره</button>
  </div>
  <div class="foot">اجرای محلی · بدون ارسال متن به اینترنت</div>
</div>
<script>
  function enDigits(s){return s.replace(/[۰-۹]/g,function(d){return String(d.charCodeAt(0)-0x06F0);});}
  var busy=false, hasAudio=false;
  function speak(){
    var t=document.getElementById('t').value.trim();
    if(!t||busy)return;
    busy=true;hasAudio=false;
    document.getElementById('go').disabled=true;
    document.getElementById('postrow').style.display='none';
    var st=document.getElementById('status');
    st.innerHTML='<span class="spin"></span> در حال ساخت صدا…';
    window.ipc.postMessage('text:'+t);
  }
  function saveIt(){busy=false;window.ipc.postMessage('save');}
  window.__gooyaResult=function(ok,msg){
    var st=document.getElementById('status');
    if(ok){
      if(msg.indexOf('ذخیره')===0){
        st.textContent=msg;document.getElementById('postrow').style.display='flex';
        busy=false;document.getElementById('go').disabled=false;return;
      }
      st.textContent='آماده شد';
      hasAudio=true;
      document.getElementById('postrow').style.display='flex';
    }else{st.textContent='خطا: '+msg;}
    busy=false;document.getElementById('go').disabled=false;
  };
  window.__gooyaPaste=function(s){
    if(!s)return;
    var t=document.activeElement;
    if(t&&(t.tagName==='TEXTAREA'||t.tagName==='INPUT'||t.isContentEditable)){
      t.focus();
      if(t.setSelectionRange)t.setSelectionRange(t.selectionStart,t.selectionEnd);
      document.execCommand('insertText',false,s);
    }
  };
  document.addEventListener('keydown',function(e){
    if(e.key==='Enter'&&!e.shiftKey){speak();return;}
    var mod=e.metaKey||e.ctrlKey;
    if(!mod)return;
    var k=(e.key||'').toLowerCase();
    var t=document.activeElement;
    var isField=t&&(t.tagName==='TEXTAREA'||t.tagName==='INPUT'||t.isContentEditable);
    if(k==='a'){
      if(isField){t.focus();t.select();e.preventDefault();}
      return;
    }
    if(k==='c'||k==='x'){
      if(isField){document.execCommand(k==='c'?'copy':'cut');}
      return;
    }
    if(k==='v'){
      if(!isField)return;
      e.preventDefault();
      window.ipc.postMessage('paste');
      return;
    }
    if(k==='z'){
      e.preventDefault();
      document.execCommand(e.shiftKey?'redo':'undo');
      return;
    }
    if(k==='y'){
      e.preventDefault();
      document.execCommand('redo');
      return;
    }
  });
</script>
</body>
</html>
"#;