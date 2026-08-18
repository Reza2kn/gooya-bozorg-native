fn main() {
    #[cfg(windows)]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("../assets/icon/Gooya.ico");
        if let Err(error) = res.compile() {
            eprintln!("winres: failed to embed icon: {error}");
            std::process::exit(1);
        }
    }
}
