import re

def fix_macro_spacing(text, target_macros, known_macros=None):
    if known_macros is None:
        known_macros = set()

    for macro in target_macros:
        pattern = re.escape(macro) + r'([a-zA-Z])(?![a-zA-Z])'

        def replace(m, macro=macro):
            letter = m.group(1)
            if (macro + letter) in known_macros:
                return m.group(0)
            return macro + ' ' + letter

        text = re.sub(pattern, replace, text)

    return text

def try_fix_macro_spacing_in_markdown(text, debug: bool = False) -> str:
    """
    只处理 \\( ... \\) 包裹的公式部分，普通文本不动。
    """
    
    known_macros = {r'\top', r'\int', r'\inf'}
    target_macros = [r'\cong', r'\to', r'\times', r'\subset', r'\in']
    
    result = []
    # 按 \\( ... \\) 分割，奇数 index 为公式内容
    parts = re.split(r'(\\\(.*?\\\))', text, flags=re.DOTALL)

    for i, part in enumerate(parts):
        if part.startswith(r'\(') and part.endswith(r'\)'):
            # 只处理公式内部内容，保留首尾定界符
            inner = part[2:-2]
            fixed_inner = fix_macro_spacing(inner, target_macros, known_macros)
            result.append(r'\(' + fixed_inner + r'\)')
        else:
            result.append(part)

    new_text = ''.join(result)
    
    if debug and new_text != text:
        print(f"Fixed equation delimeters from: {text} to: {new_text}")
    return new_text

if __name__ == "__main__":
    text = "abcdf \\( a \\timesX b \\) asdadfads"
    print(try_fix_macro_spacing_in_markdown(text))