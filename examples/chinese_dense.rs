use araea_wordcloud::{ColorScheme, WordCloudBuilder, WordInput};
use rand::Rng;
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = generate_dense_chinese_data();
    println!("Generated {} Chinese words.", words.len());

    let start = Instant::now();

    let width = 1200;
    let height = 800;

    println!("Building word cloud ({}x{})...", width, height);

    let scheme = ColorScheme::Default;
    let wordcloud = WordCloudBuilder::new()
        .size(width, height)
        .color_scheme(scheme)
        .background(scheme.background_color())
        .padding(2)
        .word_spacing(2.0)
        .angles(vec![0.0, 90.0])
        .font_size_range(10.0, 100.0)
        .seed(2025)
        .build(&words)?;

    let output_path = "output_chinese_dense.png";
    fs::write(output_path, wordcloud.to_png(2.0)?)?;

    println!("Success! Time elapsed: {:?}", start.elapsed());
    println!("Saved to {}", output_path);

    Ok(())
}

fn generate_dense_chinese_data() -> Vec<WordInput> {
    let mut words = Vec::new();
    let mut rng = rand::rng();

    // Group A: 核心超大词 (权重 80-100)
    let core_keywords = vec![
        "Rust编程",
        "高性能",
        "内存安全",
        "WebAssembly",
        "系统级",
        "无GC",
        "所有权",
        "生命周期",
        "借用检查",
        "并发模型",
    ];
    for w in core_keywords {
        words.push(WordInput::new(w, rng.random_range(80.0..100.0)));
    }

    let concepts = vec![
        "Cargo",
        "Crates.io",
        "Tokio",
        "Actix",
        "Serde",
        "Diesel",
        "Async/Await",
        "Trait",
        "Struct",
        "Enum",
        "Pattern Matching",
        "Zero-cost",
        "安全性",
        "跨平台",
        "嵌入式",
        "网络服务",
        "命令行工具",
        "错误处理",
        "Result",
        "Option",
    ];
    for w in concepts {
        words.push(WordInput::new(w, rng.random_range(50.0..80.0)));
    }

    let common = vec![
        "编译",
        "运行",
        "测试",
        "文档",
        "模块",
        "函数",
        "闭包",
        "迭代器",
        "集合",
        "字符串",
        "指针",
        "引用",
        "智能指针",
        "Box",
        "Rc",
        "Arc",
        "Mutex",
        "Channel",
        "Future",
        "Stream",
        "宏定义",
        "属性",
        "泛型",
        "Community",
        "Foundation",
        "Docs",
        "Book",
        "Examples",
        "Tutorial",
    ];
    for _ in 0..3 {
        for w in &common {
            words.push(WordInput::new(*w, rng.random_range(30.0..50.0)));
        }
    }

    let fillers = vec![
        "代码", "逻辑", "数据", "接口", "实现", "调用", "返回", "参数", "类型", "变量", "常量",
        "静态", "动态", "链接", "库", "包", "依赖", "版本", "发布", "构建", "优化", "调试", "日志",
        "监控", "设计", "架构", "模式", "算法", "结构", "效率", "速度", "稳定", "扩展", "维护",
        "重构", "迁移", "学习", "曲线", "入门", "精通",
    ];

    for _ in 0..10 {
        for w in &fillers {
            words.push(WordInput::new(*w, rng.random_range(10.0..30.0)));
        }
    }

    words
}
