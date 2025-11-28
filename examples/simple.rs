use araea_wordcloud::generate;
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    let words = vec![
        ("Rust", 100.0),
        ("Performance", 80.0),
        ("Safety", 70.0),
        ("Concurrency", 60.0),
        ("Fast", 50.0),
        ("Memory", 45.0),
        ("Efficient", 40.0),
        ("Reliable", 35.0),
        ("Community", 30.0),
        ("Cargo", 25.0),
        ("Crates", 20.0),
        ("Macro", 15.0),
    ];

    println!("Generating word cloud with {} words...", words.len());

    let wordcloud = generate(&words)?;

    let png_data = wordcloud.to_png(2.0)?;
    fs::write("output_simple.png", png_data)?;

    let svg_data = wordcloud.to_svg();
    fs::write("output_simple.svg", svg_data)?;

    println!("Done! Saved to output_simple.png and output_simple.svg");
    println!("Time elapsed: {:?}", start.elapsed());

    Ok(())
}
