def try_fix_equation_delimeters(latex: str, debug: bool = False) -> str:
    
    new_latex = latex.strip()
    if new_latex[:2] == "\\[":
        new_latex = new_latex[2:]
    if new_latex[-2:] == "\\]":
        new_latex = new_latex[:-2]
    new_latex = new_latex.strip()
    
    if debug and new_latex != latex:
        print(f"Fixed equation delimeters from: {latex} to: {new_latex}")
    return new_latex


if __name__ == "__main__":
    latex = "\\[a \\coloneqq b\\]"
    print(try_fix_equation_delimeters(latex))
