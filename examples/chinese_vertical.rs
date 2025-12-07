use araea_wordcloud::{WordCloudBuilder, WordInput};
use rand::Rng;
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = generate_dense_chinese_data();
    println!("Generated {} Chinese words.", words.len());

    let start = Instant::now();

    // 构建词云
    let wordcloud = WordCloudBuilder::new()
        // 设置画布大小
        .size(1000, 800)
        // 设置字体大小范围
        .font_size_range(20.0, 100.0)
        // 设置角度包含 0度 和 90度/-90度
        .angles(vec![0.0, -90.0])
        // 开启竖排正写功能
        // 开启后，凡是 90/-90 度的词，文字方向不会倒转，而是变为从上到下的竖排
        .vertical_writing(true)
        .build(&words)?;

    // 输出 PNG
    let output_path = "output_chinese_vertical.png";
    fs::write(output_path, wordcloud.to_png(2.0)?)?;

    let output_svg_path = "output_chinese_vertical.svg";
    fs::write(output_svg_path, wordcloud.to_svg())?;

    println!("Success! Time elapsed: {:?}", start.elapsed());
    println!("Saved to {} and {}", output_path, output_svg_path);

    Ok(())
}

fn generate_dense_chinese_data() -> Vec<WordInput> {
    let mut words = Vec::new();
    let mut rng = rand::rng();

    // 精简后的核心关键词，约10个
    let core_keywords = vec![
        "Rust编程",
        "高性能",
        "内存安全",
        "WebAssembly",
        "系统级",
        "所有权",
        "生命周期",
        "借用检查",
        "并发模型",
        "异步",
    ];
    for w in &core_keywords {
        words.push(WordInput::new(*w, rng.random_range(85.0..=100.0)));
    }

    // 精简后的概念词，约20个
    let concepts = vec![
        "Cargo",
        "Tokio",
        "Serde",
        "Trait",
        "Struct",
        "Enum",
        "Pattern Matching",
        "安全性",
        "跨平台",
        "嵌入式",
        "网络服务",
        "命令行工具",
        "错误处理",
        "Result",
        "Option",
        "Async/Await",
        "Crates.io",
        "Actix",
        "泛型",
        "宏定义",
    ];
    for w in &concepts {
        words.push(WordInput::new(*w, rng.random_range(55.0..=80.0)));
    }

    // 常用词，约20个，不重复多次
    let common = vec![
        "编译",
        "运行",
        "测试",
        "文档",
        "模块",
        "函数",
        "闭包",
        "迭代器",
        "字符串",
        "引用",
        "智能指针",
        "Box",
        "Rc",
        "Arc",
        "Mutex",
        "Channel",
        "Future",
        "Stream",
        "设计",
        "架构",
    ];
    for w in &common {
        words.push(WordInput::new(*w, rng.random_range(40.0..=55.0)));
    }

    words
}
