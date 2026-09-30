"""Reference list for the Springer (IJIS) version: reads ../paper_ijis/refs.bib and writes
../paper_ijis/references.tex (thebibliography) in the Springer basic numbered style of the journal's
submission guidelines: entries numbered in order of first citation, authors as "Surname, I.", full journal
titles, and DOIs as full https://doi.org links.  Usage: python make_springer_refs.py"""
import os, re, sys

P = os.environ.get("PAPER", "../paper_ijis")
FULL = {  # journal abbreviations in refs.bib -> full titles (guidelines: full title when unsure of the LTWA form)
    "IEEE Trans. Inf. Forensics Security": "IEEE Transactions on Information Forensics and Security",
    "Pattern Recognit.": "Pattern Recognition", "Pattern Recognit. Lett.": "Pattern Recognition Letters",
    "Optim. Eng.": "Optimization and Engineering", "Mach. Learn.": "Machine Learning",
    "J. Global Optim.": "Journal of Global Optimization", "IEEE Trans. Signal Process.": "IEEE Transactions on Signal Processing",
    "USSR Comput. Math. Math. Phys.": "USSR Computational Mathematics and Mathematical Physics", "TOP": "TOP",
    "Scand. J. Statist.": "Scandinavian Journal of Statistics",
    "Philos. Trans. Roy. Soc. London A": "Philosophical Transactions of the Royal Society of London, Series A",
    "Pattern Anal. Appl.": "Pattern Analysis and Applications", "Nature Methods": "Nature Methods", "Mathematics": "Mathematics",
    "Math. Program.": "Mathematical Programming", "Math. Program. Comput.": "Mathematical Programming Computation",
    "Manage. Sci.": "Management Science", "J. Cloud Comput.": "Journal of Cloud Computing",
    "Int. J. Comput. Commun. Control": "International Journal of Computers Communications \\& Control",
    "IEEE Trans. Pattern Anal. Mach. Intell.": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "IEEE Trans. Dependable Secure Comput.": "IEEE Transactions on Dependable and Secure Computing",
    "IEEE Trans. Biometrics, Behav., Identity Sci.": "IEEE Transactions on Biometrics, Behavior, and Identity Science",
    "IEEE Trans. Artif. Intell.": "IEEE Transactions on Artificial Intelligence",
    "IEEE Trans. Aerosp. Electron. Syst.": "IEEE Transactions on Aerospace and Electronic Systems",
    "Econometrica": "Econometrica", "Biometrika": "Biometrika",
    "Comput. Vis. Image Understand.": "Computer Vision and Image Understanding", "Appl. Intell.": "Applied Intelligence",
    "J. Vis. Lang. Comput.": "Journal of Visual Languages \\& Computing", "IET Biometrics": "IET Biometrics",
    "IEEE Trans. Image Process.": "IEEE Transactions on Image Processing", "Mach. Intell. Res.": "Machine Intelligence Research",
}
BOOKTITLE = {
    "Proc. SPIE 5404, Biometric Technology for Human Identification": "Biometric Technology for Human Identification, Proceedings of SPIE, vol.~5404",
    "Proc. IEEE Workshop Biometric Meas. Syst. Secur. Med. Appl. (BIOMS)": "2010 IEEE Workshop on Biometric Measurements and Systems for Security and Medical Applications (BIOMS)",
    "Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)": "2013 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)",
    "Proc. IEEE Int. Workshop Inf. Forensics Secur. (WIFS)": "2010 IEEE International Workshop on Information Forensics and Security (WIFS)",
}
PUBLISHER = {"Cambridge Univ. Press": "Cambridge University Press"}


def parse_bib(txt):
    out = {}
    for typ, key, body in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", txt, re.S):
        f = {}
        for m in re.finditer(r"(\w+)\s*=\s*\{", body):
            i, depth = m.end(), 1
            while depth:
                depth += {"{": 1, "}": -1}.get(body[i], 0); i += 1
            f[m.group(1).lower()] = re.sub(r"\s+", " ", body[m.end():i - 1]).strip()
        out[key.strip()] = (typ.lower(), f)
    return out


def split_top(s, sep):
    """split s at `sep` outside braces"""
    parts, depth, cur, i = [], 0, "", 0
    while i < len(s):
        if s[i] == "{": depth += 1
        if s[i] == "}": depth -= 1
        if depth == 0 and s.startswith(sep, i):
            parts.append(cur); cur = ""; i += len(sep); continue
        cur += s[i]; i += 1
    return parts + [cur]


def initials(given):
    out = []
    for tok in given.split():
        subs = tok.split("-")
        ini = "-".join(re.sub(r"^\{?\\?[^A-Za-z{]*\{?", "", t)[:1] + "." for t in subs if t)
        out.append(ini)
    return "".join(out)


def fmt_name(n):
    n = n.strip()
    if n == "others": return "et al."
    if n.startswith("{") and n.endswith("}") and n.count("{") == 1: return n[1:-1]          # corporate author
    if "," in n:
        last, given = [x.strip() for x in n.split(",", 1)]
    else:
        toks = split_top(n, " "); last, given = toks[-1], " ".join(toks[:-1])
    last = last[1:-1] if last.startswith("{") and last.endswith("}") and not last.startswith("{\\") else last
    return f"{last}, {initials(given)}" if given else last


def authors(s, ed=False):
    names = [fmt_name(x) for x in split_top(s, " and ")]
    txt = ", ".join(names)
    return txt


def clean_title(t):
    prev = None
    while prev != t:   # drop case-protecting braces that are not part of an accent command
        prev = t; t = re.sub(r"(?<![\\'`^\"~=.a-zA-Z])\{([^{}\\]*)\}", r"\1", t)
    return t.replace("---", " -- ")


def doi(f):
    return f" \\url{{https://doi.org/{f['doi']}}}" if "doi" in f else ""


def entry(key, typ, f):
    a = authors(f["author"]) if "author" in f else ""
    t = clean_title(f.get("title", ""))
    y = f.get("year", "")
    if typ == "article":
        j = FULL[f["journal"]]
        vol = f.get("volume", ""); num = f"({f['number']})" if "number" in f else ""; pg = f.get("pages", "")
        src = j + (f" {vol}{num}" if vol else "") + (f", {pg}" if pg else "")
        return f"{a}: {t}. {src} ({y}).{doi(f)}"
    if typ == "inproceedings":
        return f"{a}: {t}. In: {BOOKTITLE[f['booktitle']]}, pp.~{f['pages']} ({y}).{doi(f)}"
    if typ == "incollection":
        ed = authors(f["editor"]); ser = f", {f['series']}, vol.~{f['volume']}" if "series" in f else ""
        city = f.get("address", "").split(",")[0]
        return f"{a}: {t}. In: {ed} (ed.) {f['booktitle']}{ser}, pp.~{f['pages']}. {f['publisher']}, {city} ({y}).{doi(f)}"
    if typ == "book":
        city = f.get("address", "").split(",")[0]
        return f"{a}: {t}. {PUBLISHER.get(f['publisher'], f['publisher'])}, {city} ({y}).{doi(f)}"
    if typ == "techreport":
        return f"{a}: {t}. Tech. Rep. {f['number']}, {f['institution']} ({y}).{doi(f)}"
    if typ == "mastersthesis":
        return f"{a}: {t}. Master's thesis, Department of Information Management, National Taiwan University, Taipei, Taiwan ({y}).{doi(f)}"
    if typ == "misc" and f.get("howpublished", "").startswith("ISO/IEC"):
        std = f["howpublished"].split(",")[0]
        return (f"{a}: {std} {t}, 2nd edn. International Organization for Standardization, Geneva ({y})")
    if typ == "misc" and key == "bssr1":
        url = re.search(r"\\url\{([^}]*)\}", f["howpublished"]).group(1)
        acc = f.get("note", "").replace("Accessed: ", "")
        m = re.match(r"(\w+)\. (\d+), (\d+)", acc)
        mon = {"Sep": "September"}.get(m.group(1), m.group(1)) if m else ""
        return f"{a}: {t}. \\url{{{url}}}. Accessed {m.group(2)} {mon} {m.group(3)}"
    raise ValueError(f"unhandled entry {key} ({typ})")


def cite_order(main):
    seen, order = set(), []
    def walk(path):
        s = open(path).read()
        s = re.sub(r"(?<!\\)%.*", "", s)
        for m in re.finditer(r"\\(?:input|inputtab)\{([^}]*)\}|\\cite(?:\[[^\]]*\])?\{([^}]*)\}", s):
            if m.group(1):
                p = os.path.join(P, m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
                if os.path.exists(p) and not p.endswith("references.tex"): walk(p)
            else:
                for k in m.group(2).split(","):
                    k = k.strip()
                    if k not in seen: seen.add(k); order.append(k)
    walk(main)
    return order


if __name__ == "__main__":
    bib = parse_bib(open(f"{P}/refs.bib").read())
    order = cite_order(f"{P}/main.tex")
    missing = [k for k in order if k not in bib]; unused = [k for k in bib if k not in order]
    if missing: sys.exit(f"cited but not in refs.bib: {missing}")
    lines = ["% Generated by code/make_springer_refs.py from refs.bib -- do not edit by hand.",
             f"\\begin{{thebibliography}}{{{len(order)}}}"]
    for k in order:
        typ, f = bib[k]
        lines.append(f"\\bibitem{{{k}}} {entry(k, typ, f)}")
    lines.append("\\end{thebibliography}")
    open(f"{P}/references.tex", "w").write("\n".join(lines) + "\n")
    print(f"{len(order)} references written; not cited: {unused}")
