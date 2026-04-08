import re

def try_move_underscores_outside(text: str, debug: bool = False) -> str:
    def process_match(m):
        inner = m.group(1)
        # 按 3+ 个连续下划线分割
        parts = re.split(r'(_{3,})', inner)

        if len(parts) == 1:
            return m.group(0)  # 没有下划线，不处理

        result = []
        for part in parts:
            if re.fullmatch(r'_{3,}', part):
                result.append(part)          # 下划线放到外面，作为纯文本
            elif part.strip():
                result.append(r'\(' + part + r'\)')  # 剩余公式内容重新包裹

        return ' '.join(result)
    
    new_text = re.sub(r'\\\((.+?)\\\)', process_match, text, flags=re.DOTALL)
    
    if debug and new_text != text:
        print(f"Fixed equation delimeters from: {text} to: {new_text}")
    
    return new_text

if __name__ == "__main__":
    # text = r"(13) 设  \( z = \left(\frac{y}{x}\right)^{\frac{x}{y}} \) ，则  \( \frac{\partial z}{\partial x}\bigg|_{(1,2)} = ____ \) ."
    text = r"xxx = \( \sigma ___ = \lambda \)"
    
    print(try_move_underscores_outside(text))