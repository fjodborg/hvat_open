use super::*;

#[test]
fn test_make_url_safe() {
    assert_eq!(make_url_safe("hello"), "hello");
    assert_eq!(make_url_safe("hello world"), "hello_world");
    assert_eq!(make_url_safe("path/to/file.png"), "path_to_file_png");
    assert_eq!(make_url_safe("file-name_123"), "file-name_123");
    assert_eq!(make_url_safe(""), "");
    assert_eq!(make_url_safe("a/b/c"), "a_b_c");
    // Unicode alphanumeric characters are preserved (is_alphanumeric is Unicode-aware)
    assert_eq!(make_url_safe("日本語"), "日本語");
}

#[test]
fn test_make_url_safe_preserves_alphanumeric() {
    let input = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    assert_eq!(make_url_safe(input), input);
}

#[test]
fn test_make_url_safe_special_chars() {
    assert_eq!(make_url_safe("file.png"), "file_png");
    assert_eq!(make_url_safe("path\\to\\file"), "path_to_file");
    assert_eq!(make_url_safe("file name (1).jpg"), "file_name__1__jpg");
}
