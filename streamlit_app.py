import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import re
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid
import textwrap
from tempfile import NamedTemporaryFile
import os
import time

try:
    import fitz
    PDF_OK = True
except Exception:
    PDF_OK = False

HR_USERS = {
    "hr01": "1234",
    "admin": "admin123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

def login_page():
    st.markdown("## 🔐 HR Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login", type="primary"):
        if u in HR_USERS and HR_USERS[u] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.success("เข้าสู่ระบบสำเร็จ")
            st.rerun()
        else:
            st.error("Username หรือ Password ไม่ถูกต้อง")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

st.set_page_config(page_title="ระบบคัดกรองใบสมัครงานด้วย AI", page_icon="📄", layout="wide")

if not st.session_state.logged_in:
    login_page()
    st.stop()

BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if "pass_cut" not in st.session_state:
    st.session_state["pass_cut"] = 0.5

# ---------- helpers ----------
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def thai_tokenize(text):
    text = normalize_text(text)
    if not text: return []
    try:
        from pythainlp import word_tokenize
        return [t for t in word_tokenize(text, keep_whitespace=False) if t.strip()]
    except Exception:
        return text.split()

def encode_education(level):
    if not isinstance(level, str): return 0
    s = normalize_text(level)
    mapping = {
        "มัธยม":0,"ม.6":0,"ปวช":0,"hs":0,"high school":0,
        "ปวส":1,"อนุปริญญา":1,"bachelor":1,"ปริญญาตรี":1,"ba":1,"b.sc":1,"b.eng":1,
        "master":2,"ปริญญาโท":2,"m.sc":2,"mba":2,
        "phd":3,"doctorate":3,"ปริญญาเอก":3
    }
    return mapping.get(s, mapping.get(s.title(), 0))

def split_skills(skills):
    if not isinstance(skills, str): return []
    return [s.strip().lower() for s in re.split(r"[,/;|\n]", skills) if s.strip()]

def concat_text_df(df, fields):
    cols = [c for c in fields if c in df.columns]
    if not cols: return pd.Series([""] * len(df))
    return df[cols].apply(lambda r: " ".join((str(r[c]).lower() if isinstance(r[c], str) else str(r[c])) for c in cols), axis=1)

def extract_text_from_pdf(uploaded_file):
    if not PDF_OK: return ""
    try:
        uploaded_file.seek(0)
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            if doc.is_encrypted:
                try: doc.authenticate("")
                except Exception: pass
            for page in doc:
                text += page.get_text("text")
        return text.strip()
    except Exception as e:
        st.warning(f"อ่าน PDF ไม่สำเร็จ: {e}")
        return ""

class EducationEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X["education_level"].apply(encode_education).astype(float).values.reshape(-1, 1)

def combine_exp_func(arr):
    if hasattr(arr, "__dataframe__") or isinstance(arr, pd.DataFrame):
        y = pd.to_numeric(arr.get("years_experience", 0), errors="coerce").fillna(0)
        m = pd.to_numeric(arr.get("months_experience", 0), errors="coerce").fillna(0)
        return (y + m/12.0).values.reshape(-1,1)
    a = np.asarray(arr)
    y = pd.to_numeric(pd.Series(a[:,0]), errors="coerce").fillna(0)
    m = pd.to_numeric(pd.Series(a[:,1] if a.shape[1]>1 else 0), errors="coerce").fillna(0)
    return np.asarray(y+m/12.0).reshape(-1,1)

def safe_load_joblib(p):
    try: return joblib.load(p)
    except Exception: return None

def atomic_joblib_dump(obj, path, compress=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile(delete=False, dir=os.path.dirname(path), suffix=".tmp") as t:
        tmp = t.name
    try:
        joblib.dump(obj, tmp, compress=compress)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

# email clean/validate
WS_EDGES = re.compile(r'^\s+|\s+$', flags=re.UNICODE)
INVISIBLE = {"\u200b", "\ufeff"}
WIDE_SPACES = {"\u00a0", "\u202f", "\u2007", "\u2009", "\u200a", "\u205f", "\u3000"}
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
BLOCKED_DEMO_DOMAINS = {"example.com","test.com","invalid","example.org","example.net"}

def clean_email(s):
    if not isinstance(s, str): return ""
    for ch in INVISIBLE: s = s.replace(ch, "")
    for ch in WIDE_SPACES: s = s.replace(ch, " ")
    return WS_EDGES.sub("", s)

def is_valid_email(s):
    s = clean_email(s)
    if not s or not EMAIL_RE.match(s): return False
    return s.split("@")[-1].lower() not in BLOCKED_DEMO_DOMAINS

# ---------- configs ----------
DEPARTMENTS_CFG = {
    "Audit": {"must_have_skills":["ความรู้บัญชีและภาษี","การวิเคราะห์งบการเงิน","excel ขั้นสูง","ความละเอียดรอบคอบ"],
              "nice_to_have_skills":["ระบบ erp","สื่อสารกับผู้บริหาร","กฎหมายธุรกิจ"],
              "min_years_experience":1,"min_education_level":"ปริญญาตรี",
              "knockout_phrases":["ไม่ชอบงานซ้ำ","ไม่ถนัดตัวเลข","ชอบทำงานสร้างสรรค์มากกว่าเอกสาร"],
              "age_range":[25,35],
              "activity_weights":{"โครงการบัญชี":0.3,"แข่งขันแผนธุรกิจ":0.3,"แข่งแผนธุรกิจ":0.3,"business plan":0.3,"ฝึกงานบริษัทบัญชี":0.4,"ฝึกงานสำนักงานบัญชี":0.4,"internship accounting firm":0.4,"internship audit":0.4},
              "max_activity_bonus":0.8},
    "Purchasing China": {"must_have_skills":["การต่อรองราคา","ความเข้าใจระบบนำเข้า","ภาษาจีน","ภาษาอังกฤษ","ความละเอียดเอกสาร"],
              "nice_to_have_skills":["โลจิสติกส์","erp","วิเคราะห์ต้นทุน"],
              "min_years_experience":2,"min_education_level":"ปริญญาตรี",
              "knockout_phrases":["ไม่ถนัดติดต่อคนต่างชาติ","ไม่ชอบต่อรอง","ไม่เคยทำงานเอกสารนำเข้า"],
              "age_range":[25,38],
              "activity_weights":{"สื่อสารกับต่างชาติ":0.4,"เป็นล่าม":0.5,"ล่ามจีน":0.5,"แลกเปลี่ยนจีน":0.5,"exchange china":0.5,"นักเรียนแลกเปลี่ยนจีน":0.5,"ล่ามภาษาจีน":0.5},
              "max_activity_bonus":1.0},
    "Accounting": {"must_have_skills":["ทำบัญชีรายวัน","ทำบัญชีรายเดือน","งบการเงิน","ภาษีซื้อ","ภาษีขาย","excel","express","sap"],
              "nice_to_have_skills":["power bi","วิเคราะห์ต้นทุน","ภาษาอังกฤษ"],
              "min_years_experience":1,"min_education_level":"ปริญญาตรี",
              "activity_weights":{"ฝึกงานสำนักงานบัญชี":0.3,"internship accounting firm":0.3},
              "age_range":[22,35],"max_activity_bonus":0.6},
    "Human Resources": {"must_have_skills":["สรรหา","สัมภาษณ์","คำนวณเงินเดือน","กฎหมายแรงงาน","สื่อสารดี"],
              "nice_to_have_skills":["hrm","จิตวิทยา","od"],
              "min_years_experience":1,"min_education_level":"ปริญญาตรี",
              "knockout_phrases":["ไม่ชอบเจอคนเยอะ","ไม่ชอบงานเอกสาร","ต้องการเลื่อนตำแหน่งเร็ว"],
              "age_range":[24,40],
              "activity_weights":{"ชมรมผู้นำ":0.4,"ผู้นำ":0.4,"อาสา":0.4,"จิตอาสา":0.4,"จัดค่าย":0.4,"ค่าย":0.3,"camp organizer":0.4},
              "max_activity_bonus":1.2},
    "Programmer": {"must_have_skills":["php","python","javascript","sql","debugging","agile"],
              "nice_to_have_skills":["ux/ui","cloud","aws","azure","cybersecurity"],
              "min_years_experience":0,"min_education_level":"ปริญญาตรี",
              "knockout_phrases":["ไม่ชอบทำงานเป็นทีม","ไม่ถนัดแก้โค้ดคนอื่น","ไม่อยากเรียนรู้เทคโนโลยีใหม่"],
              "age_range":[21,35],
              "activity_weights":{"โปรเจกต์":0.6,"project":0.6,"hackathon":0.6,"แข่งขันเขียนโปรแกรม":0.6,"open source":0.6,"github":0.5,"สร้างแอป":0.6,"web app":0.5,"mobile app":0.5},
              "max_activity_bonus":1.2},
    "Marketing": {"must_have_skills":["วางแผนการตลาด","เขียนคอนเทนต์","facebook ads","google ads","วิเคราะห์ข้อมูล"],
              "nice_to_have_skills":["ถ่ายภาพ","ตัดต่อ","seo","sem","analytics"],
              "min_years_experience":1,"min_education_level":"ปริญญาตรี",
              "knockout_phrases":["ไม่ชอบใช้โซเชียล","ไม่เก่งตัวเลข","เน้นความรู้สึกมากกว่าผลลัพธ์"],
              "age_range":[22,35],
              "activity_weights":{"ทำเพจ":0.5,"เพจ":0.3,"ทำคอนเทนต์":0.5,"content creator":0.5,"โปรโมตงานมหาลัย":0.5,"ประชาสัมพันธ์":0.4,"ทำสื่อ":0.4,"admin page":0.4},
              "max_activity_bonus":1.2},
    "Delivery": {"must_have_skills":["รู้เส้นทางขนส่ง","จัดตารางส่งสินค้า","เอกสารขนส่ง"],
              "nice_to_have_skills":["gps","ใบขับขี่","ระบบคลังสินค้า"],
              "min_years_experience":1,"min_education_level":"ปวช",
              "activity_weights":{"พาร์ทไทม์ขนส่ง":0.3,"part-time delivery":0.3,"rider":0.3,"ฝึกงานโลจิสติกส์":0.3},
              "age_range":[22,40],"max_activity_bonus":0.6},
    "Warehouse": {"must_have_skills":["ตรวจนับสินค้า","บันทึกสต็อก","โปรแกรมคลังสินค้า"],
              "nice_to_have_skills":["โฟล์กลิฟต์","fifo","ความปลอดภัยในการทำงาน"],
              "min_years_experience":1,"min_education_level":"ปวช",
              "activity_weights":{"โครงการจัดการคลัง":0.2,"จัดการสินค้า":0.2,"ฝึกงานคลังสินค้า":0.3,"inventory project":0.2},
              "age_range":[20,40],"max_activity_bonus":0.5}
}

DEFAULT_RULES = {
    "rule_weights":{"exp_under":-1.0,"exp_meet":0.2,"edu_under":-1.0,"must_missing":-1.5,"must_all":0.5,
                    "knockout":-2.0,"nice_each":0.3,"training_ctx":-0.5,"low_exp_skill":-0.3,"age_out":-1.0,"age_in":0.2},
    "training_indicators":["อบรม","สัมมนา","เวิร์กชอป","workshop","training","คอร์ส","course","certificate","certification"],
    "activity_weights":{},"max_activity_bonus":0.0,"default_age_range":[18,60],
    "global_knockout":{"hard":["ด่าบริษัทเก่า","ด่าหัวหน้าเก่า","โกง","ทุจริต","ปลอมเอกสาร","เปิดเผยความลับบริษัท","เกลียดลูกค้า","ก้าวร้าว","ใช้ความรุนแรง"],
                       "soft":["ไม่ชอบทำงานเป็นทีม","ไม่ชอบเจอคนเยอะ","ไม่ถนัดตัวเลข","มาสายบ่อย","ไม่พร้อมทำงานล่วงเวลา","ไม่พร้อมทำงานวันหยุด","ไม่ชอบงานซ้ำ ๆ","ไม่อยากเรียนรู้เพิ่ม","ไม่ถนัดแก้โค้ดคนอื่น","ไม่ละเอียด","ไม่ละเอียดรอบคอบ"],
                       "review":["ลาออกเพราะขัดแย้งส่วนตัว","ไม่ถูกกับหัวหน้าเก่า","ลาออกทันที","ทำงานไม่นาน"]},
    "global_knockout_weights":{"hard":-2.0,"soft":-1.0,"review":0.0},
    "global_knockout_cap":-2.5,
    "global_mitigation":{"whitelist_phrases":["เคยไม่ถนัดตัวเลข แต่กำลังพัฒนา","เคยไม่ถนัดตัวเลข แต่ฝึกจนทำได้","เคยไม่ชอบทำงานเป็นทีม แต่ปรับตัวได้","แม้ไม่มีประสบการณ์ แต่เรียนรู้เร็ว","ยอมรับข้อผิดพลาดและปรับปรุง"],
                         "mitigate_terms":["แต่","กำลังพัฒนา","กำลังเรียนรู้","พัฒนา","ปรับปรุง","ตั้งใจฝึก","เรียนเพิ่ม","จนทำได้","แก้ไข"],
                         "mitigation_factor":0.5,"context_window":50}
}
DEFAULT_BASE = {"text_fields":["resume_text","cover_letter_text","skills","activities"],"id_field":"candidate_id","label_field":"label"}

ui_css = """
<style>
:root { --brand:#0EA5E9; --brand2:#22D3EE; }
.hero{background:linear-gradient(135deg,#0284C7,#22D3EE);color:#fff;padding:38px 28px 30px 28px;border-radius:16px;box-shadow:0 10px 25px rgba(14,165,233,.25);margin-bottom:18px;}
.kpi{background:#fff;border:1px solid #e6f4ff;border-radius:14px;padding:14px 16px;box-shadow:0 4px 14px rgba(14,165,233,.08)}
.kpi h4{margin:0 0 8px 0;font-size:14px;color:#0b6aa7}
.kpi .v{font-size:24px;font-weight:700}
</style>
"""
st.markdown(ui_css, unsafe_allow_html=True)
st.markdown('<div class="hero"><h3>📄 ระบบคัดกรองใบสมัครงานด้วย AI</h3></div>', unsafe_allow_html=True)

# ---------- sidebar ----------
with st.sidebar:
    st.markdown(f"👤 HR: {st.session_state.username}")
    if st.button("Logout"):
        logout()

    st.header("เลือกแผนก / การตั้งค่า")
    department = st.selectbox("เลือกแผนก", list(DEPARTMENTS_CFG.keys()))
    st.slider("เกณฑ์ผ่าน (Pass) ≥", 0.10, 0.90, step=0.01, key="pass_cut")
    pass_cut = st.session_state.pass_cut

    enable_email = st.toggle("เปิดใช้งานแจ้งผลผู้สมัครทางอีเมล", value=False)
    if enable_email:
        provider = st.selectbox("เลือกผู้ให้บริการ SMTP", ["Gmail", "Outlook/Hotmail"])
        if provider == "Gmail":
            smtp_host_default, port_default, ssl_default = "smtp.gmail.com", 587, False
        else:
            smtp_host_default, port_default, ssl_default = "smtp.office365.com", 587, False

        smtp_host = st.text_input("SMTP Host", value=smtp_host_default)
        smtp_port = st.number_input("Port", min_value=1, max_value=65535, value=port_default)
        use_ssl = st.toggle("ใช้ SSL (ถ้าเปิดให้ใช้พอร์ต 465)", value=ssl_default)
        smtp_user = st.text_input("Username (อีเมลผู้ส่ง)")
        smtp_pass = st.text_input("Password/App Password", type="password")
        from_name = st.text_input("From Name", value="HR Team")
        from_email = st.text_input("From Email (ควรตรงกับ Username)", value=smtp_user)

        subj_pass = st.text_input("หัวเรื่อง (ผ่าน)", value="[บริษัท] ผลการสมัครงาน – ผ่านการคัดกรองเบื้องต้น")
        subj_fail = st.text_input("หัวเรื่อง (ไม่ผ่าน)", value="[บริษัท] ผลการสมัครงาน – ไม่ผ่านการคัดกรอง")

        default_pass = textwrap.dedent("""\
เรียน {{name}} ({{candidate_id}})
ผลการคัดกรองตำแหน่ง {{department}} : 'ผ่าน'
คะแนนรวม: {{score}}
ทีมงานจะติดต่อกลับเพื่อแจ้งขั้นตอนถัดไป ขอบคุณครับ/ค่ะ
""").strip()
        default_fail = textwrap.dedent("""\
เรียน {{name}} ({{candidate_id}})
ผลการคัดกรองตำแหน่ง {{department}} : 'ไม่ผ่าน'
สาเหตุหลัก: {{reasons}}
ขอบคุณที่ให้ความสนใจ และหวังว่าจะมีโอกาสร่วมงานกันในอนาคตครับ/ค่ะ
""").strip()
        msg_tpl_pass = st.text_area("เทมเพลต (ผ่าน)", value=default_pass, height=140)
        msg_tpl_fail = st.text_area("เทมเพลต (ไม่ผ่าน)", value=default_fail, height=160)

        st.markdown("---")

        def _send_email(h, p, ssl_on, user, pwd, fname, femail, to_email, sub, body_text):
            msg = MIMEText(body_text, _charset="utf-8")
            msg["Subject"] = sub
            msg["From"] = formataddr((fname, femail))
            msg["To"] = to_email
            msg["Date"] = formatdate(localtime=True)
            msg["Message-Id"] = make_msgid()
            if ssl_on:
                server = smtplib.SMTP_SSL(h, int(p))
            else:
                server = smtplib.SMTP(h, int(p))
                try: server.starttls()
                except Exception: pass
            try:
                if user: server.login(user, pwd)
                refused = server.sendmail(femail, [to_email], msg.as_string())
                ok = (len(refused) == 0)
                return ok, ("" if ok else str(refused))
            finally:
                try: server.quit()
                except Exception: pass

        test_to = st.text_input("ทดสอบส่งไปที่อีเมล", value=smtp_user or "")
        if st.button("ทดสอบส่งเมล"):
            ok, info = _send_email(smtp_host, smtp_port, use_ssl, smtp_user, smtp_pass,
                                   from_name, from_email or smtp_user, clean_email(test_to),
                                   "SMTP Test • AI Screening", "ข้อความทดสอบการส่งอีเมลจากระบบ")
            if ok: st.success(f"ส่งทดสอบสำเร็จไปที่ {clean_email(test_to)}")
            else: st.error(f"ส่งทดสอบไม่สำเร็จ: refused {info}")

# ---------- ML ----------
def build_pipeline(text_fields):
    text_vect = TfidfVectorizer(tokenizer=thai_tokenize, token_pattern=None, ngram_range=(1,2), min_df=1, max_df=0.95)
    text_transformer = Pipeline(steps=[
        ("concat", FunctionTransformer(concat_text_df, kw_args={"fields": tuple(text_fields)}, validate=False)),
        ("tfidf", text_vect)
    ])
    pre = ColumnTransformer(
        transformers=[
            ("text", text_transformer, text_fields),
            ("exp", FunctionTransformer(combine_exp_func, validate=False), ["years_experience","months_experience"]),
            ("edu", EducationEncoder(), ["education_level"]),
            ("age", "passthrough", ["age"])
        ],
        remainder="drop"
    )
    clf = LogisticRegression(max_iter=300)
    return Pipeline(steps=[("pre", pre), ("clf", clf)])

def _apply_global_knockout(blob, cfg):
    gk = cfg.get("global_knockout", DEFAULT_RULES["global_knockout"])
    weights = cfg.get("global_knockout_weights", DEFAULT_RULES["global_knockout_weights"])
    cap = cfg.get("global_knockout_cap", DEFAULT_RULES["global_knockout_cap"])
    mitig = cfg.get("global_mitigation", DEFAULT_RULES["global_mitigation"])
    wlist = [normalize_text(x) for x in mitig.get("whitelist_phrases", [])]
    mterms = [normalize_text(x) for x in mitig.get("mitigate_terms", [])]
    factor = float(mitig.get("mitigation_factor", 0.5))
    ctx = int(mitig.get("context_window", 50))
    penalty = 0.0
    hits = []
    for level in ["hard","soft","review"]:
        for phrase in gk.get(level, []):
            p = normalize_text(phrase)
            if not p: continue
            pos = blob.find(p)
            if pos == -1: continue
            if any(normalize_text(w) in blob for w in wlist):
                hits.append(f"{p}~whitelist"); continue
            start = max(0, pos - ctx); end = min(len(blob), pos + len(p) + ctx)
            window = blob[start:end]
            weight = float(weights.get(level, 0.0))
            if any(t in window for t in mterms) and weight < 0:
                weight *= factor; hits.append(f"{p}:{level}*{factor:.1f}")
            else:
                hits.append(f"{p}:{level}")
            penalty += weight
    if penalty < cap: penalty = cap
    return penalty, hits

def rule_score(row, cfg):
    def _to_num(x):
        try: return float(x)
        except: return 0.0
    w = cfg.get("rule_weights", DEFAULT_RULES["rule_weights"])
    reasons, score = [], 0.0

    y = _to_num(row.get("years_experience", 0))
    m = _to_num(row.get("months_experience", 0))
    total = y + m/12.0
    if total < cfg["min_years_experience"]:
        reasons.append(f"ประสบการณ์ {total:.1f} ปี < {cfg['min_years_experience']} ปี"); score += w.get("exp_under", -1.0)
    else:
        reasons.append(f"ประสบการณ์ {total:.1f} ปี ≥ เกณฑ์"); score += w.get("exp_meet", 0.2)

    if encode_education(row.get("education_level","")) < encode_education(cfg["min_education_level"]):
        reasons.append(f"การศึกษา < {cfg['min_education_level']}"); score += w.get("edu_under", -1.0)

    skills = set(split_skills(row.get("skills","")))
    miss = [s for s in cfg["must_have_skills"] if s not in skills]
    if miss:
        reasons.append("ขาดสกิลจำเป็น: "+", ".join(miss)); score += w.get("must_missing", -1.5)
    else:
        reasons.append("ครบสกิลจำเป็น"); score += w.get("must_all", 0.5)

    blob = " ".join([normalize_text(str(row.get(f,""))) for f in cfg["text_fields"]])

    gk_pen, gk_hits = _apply_global_knockout(blob, cfg)
    if gk_hits: reasons.append("คำต้องห้ามรวม: " + ", ".join(gk_hits))
    score += gk_pen

    for phrase in cfg.get("knockout_phrases", []):
        if normalize_text(phrase) in blob:
            reasons.append(f"คำต้องห้ามตำแหน่ง: {phrase}"); score += w.get("knockout", -2.0)

    training_indicators = cfg.get("training_indicators", DEFAULT_RULES["training_indicators"])
    ind_alt = "(" + "|".join(map(re.escape, training_indicators)) + ")"
    for s in cfg.get("must_have_skills", []):
        s_esc = re.escape(s)
        ps1 = re.compile(ind_alt + r".{0,40}\b" + s_esc + r"\b")
        ps2 = re.compile(r"\b" + s_esc + r"\b.{0,40}" + ind_alt)
        if ps1.search(blob) or ps2.search(blob):
            reasons.append(f"{s}: พบในบริบทการอบรม/คอร์ส"); score += w.get("training_ctx", -0.5)

    nice = [s for s in cfg.get("nice_to_have_skills", []) if s in skills]
    if nice:
        reasons.append("มีสกิลเสริม: "+", ".join(nice)); score += w.get("nice_each", 0.3) * len(nice)

    low_exp = max(0.5, cfg.get("min_years_experience",0)/2)
    for s in cfg.get("must_have_skills", []):
        if s in skills and total < low_exp:
            reasons.append(f"มี {s} แต่ประสบการณ์รวม < {low_exp:.1f} ปี"); score += w.get("low_exp_skill", -0.3)

    age_val = _to_num(row.get("age", -1))
    age_range = cfg.get("age_range", DEFAULT_RULES.get("default_age_range", [18,60]))
    if age_val >= 0:
        if not (age_range[0] <= age_val <= age_range[1]):
            reasons.append(f"อายุ {int(age_val)} ปี อยู่นอกช่วง ({age_range[0]}–{age_range[1]})"); score += w.get("age_out", -1.0)
        else:
            reasons.append(f"อายุ {int(age_val)} ปี อยู่ในช่วงที่กำหนด"); score += w.get("age_in", 0.2)
    else:
        reasons.append("ไม่ระบุอายุ (ไม่คิดคะแนน)")

    acts = normalize_text(str(row.get("activities", "")))
    aw = cfg.get("activity_weights", DEFAULT_RULES.get("activity_weights", {}))
    max_bonus = cfg.get("max_activity_bonus", DEFAULT_RULES.get("max_activity_bonus", 0.0))
    is_training_like_activity = any(normalize_text(ind) in acts for ind in training_indicators)
    bonus = 0.0; hits=[]
    for k, wt in aw.items():
        if normalize_text(k) in acts:
            hits.append(f"{k}+{wt}"); bonus += float(wt)
    if bonus > 0:
        if is_training_like_activity:
            reasons.append("กิจกรรมมีลักษณะอบรม: ไม่ได้โบนัสกิจกรรม")
        else:
            bonus = min(bonus, max_bonus)
            reasons.append("กิจกรรม: " + ", ".join(hits) + (f" (รวม +{bonus:.1f})" if bonus>0 else "")); score += bonus

    return score, reasons

def bucket_level(final_priority, pass_cut):
    return "ผ่าน" if final_priority >= pass_cut else "ไม่ผ่าน"

# ---------- tabs ----------
tab_train, tab_screen = st.tabs(["ฝึกสอนโมเดล (CSV)", "คัดกรองใบสมัคร (CSV/PDF)"])

with tab_train:
    c1, c2 = st.columns([2,1])
    with c1:
        train_file = st.file_uploader("อัปโหลดไฟล์ CSV สำหรับฝึกสอน (ต้องมี label 0/1)", type=["csv"], key="train_csv")
    with c2:
        train_btn = st.button("เริ่มฝึกสอน", type="primary", use_container_width=True)
        if (MODEL_DIR / "model.joblib").exists():
            st.download_button("ดาวน์โหลดโมเดล", data=open(MODEL_DIR / "model.joblib","rb"),
                               file_name="model.joblib", use_container_width=True)
    if train_btn and train_file is not None:
        df = pd.read_csv(train_file)
        need = [*"resume_text cover_letter_text skills activities".split(), "years_experience", "education_level", "label"]
        miss = [c for c in need if c not in df.columns]
        if "months_experience" not in df.columns: df["months_experience"] = 0
        if "age" not in df.columns: df["age"] = -1
        if "activities" not in df.columns: df["activities"] = ""
        if miss:
            st.error(f"คอลัมน์หายไป: {miss}")
        else:
            y = df["label"]
            pipe = build_pipeline(["resume_text","cover_letter_text","skills","activities"])
            with st.spinner("กำลังฝึกสอน..."):
                pipe.fit(df, y)
                atomic_joblib_dump({"pipeline": pipe, "config": DEFAULT_BASE}, MODEL_DIR / "model.joblib", compress=3)
            st.success("ฝึกสอนสำเร็จ")

with tab_screen:
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        ftype = st.radio("รูปแบบไฟล์", ["CSV (หลายคน)", "PDF (ทีละไฟล์)"], horizontal=True)
        infile = st.file_uploader("อัปโหลดไฟล์", type=["csv"] if ftype.startswith("CSV") else ["pdf"], key="screen_file")
        if ftype.endswith("PDF") and not PDF_OK:
            st.info("ต้องติดตั้ง PyMuPDF: pip install PyMuPDF")
    with c2:
        up_model = st.file_uploader("อัปโหลดโมเดล (.joblib) หรือใช้โมเดลล่าสุด", type=["joblib"], key="mup")
        model = None
        if up_model is not None:
            try:
                model = joblib.load(up_model); st.success("โหลดโมเดลจากไฟล์อัปโหลดสำเร็จ")
            except Exception as e:
                st.error(f"โหลดโมเดลจากไฟล์ไม่สำเร็จ: {e}")
        if model is None and (MODEL_DIR / "model.joblib").exists():
            model = safe_load_joblib(MODEL_DIR / "model.joblib")
            if model is None:
                st.error("ไฟล์โมเดลล่าสุดเสีย โปรดฝึกใหม่หรืออัปโหลดไฟล์ใหม่")
        if model is None and not (MODEL_DIR / "model.joblib").exists():
            st.warning("ยังไม่มีโมเดล โปรดฝึกสอนในแท็บแรกหรืออัปโหลด .joblib")
    with c3:
        run_btn = st.button("เริ่มประเมิน", type="primary", use_container_width=True)

    def _render_template(tpl, ctx):
        out = tpl
        for k, v in ctx.items():
            out = out.replace(f"{{{{{k}}}}}", str(v))
        return out

    if run_btn and infile is not None and model is not None:
        pipe = model["pipeline"]
        if ftype.endswith("PDF"):
            pdf_text = extract_text_from_pdf(infile)
            df = pd.DataFrame([{
                "candidate_id": infile.name, "resume_text": pdf_text, "cover_letter_text": "",
                "years_experience": 0, "months_experience": 0, "education_level": "ไม่ระบุ",
                "skills": "", "activities": "", "age": -1, "email": "", "name": "ผู้สมัคร"
            }])
        else:
            df = pd.read_csv(infile)
            if "months_experience" not in df.columns: df["months_experience"] = 0
            if "age" not in df.columns: df["age"] = -1
            if "activities" not in df.columns: df["activities"] = ""
            if "email" not in df.columns: df["email"] = ""
            if "name" not in df.columns: df["name"] = "ผู้สมัคร"

        need = ["candidate_id","resume_text","cover_letter_text","skills","activities","years_experience","education_level"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            st.error(f"คอลัมน์หายไป: {miss}")
        else:
            proba = pipe.predict_proba(df)[:, 1]
            rows = []
            for i, row in df.iterrows():
                # ←← แก้เรียบร้อย: ปิดด้วย '}' ไม่ใช่ ']'
                merged_cfg = {**DEFAULT_RULES, **DEPARTMENTS_CFG[department], **DEFAULT_BASE}
                rscore, reasons = rule_score(row, merged_cfg)
                final = 0.7 * float(proba[i]) + 0.3 * (1 / (1 + np.exp(-rscore)))

                def _to_num(x):
                    try: return float(x)
                    except: return 0.0

                level = bucket_level(final, pass_cut)
                y_exp = _to_num(row.get("years_experience", 0)); m_exp = _to_num(row.get("months_experience", 0))
                total_months = int(round(y_exp * 12 + m_exp))
                y_show = total_months // 12; m_show = total_months % 12
                rows.append({
                    "candidate_id": row.get("candidate_id", i),
                    "name": row.get("name","ผู้สมัคร"),
                    "email": clean_email(row.get("email","")),
                    "experience_text": f"{int(y_show)} ปี {int(m_show)} เดือน",
                    "age": (int(row.get("age", -1)) if pd.notna(row.get("age", -1)) else -1),
                    "predicted_fit": round(float(proba[i]), 4),
                    "rule_score": round(float(rscore), 2),
                    "final_priority": round(float(final), 4),
                    "predicted_level": level,
                    "reasons": "; ".join(reasons)
                })
            out = pd.DataFrame(rows).sort_values("final_priority", ascending=False).reset_index(drop=True)
            out["email"] = out["email"].apply(clean_email)

            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="kpi"><h4>จำนวนผู้สมัคร</h4><div class="v">{len(out)}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="kpi"><h4>คะแนนสูงสุด</h4><div class="v">{out["final_priority"].max():.2f}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="kpi"><h4>ค่าเฉลี่ยคะแนน</h4><div class="v">{out["final_priority"].mean():.2f}</div></div>', unsafe_allow_html=True)

            show_only_pass = st.toggle("แสดงเฉพาะผู้ที่ผ่านเกณฑ์", value=False)
            if show_only_pass: out = out[out["predicted_level"] == "ผ่าน"].reset_index(drop=True)

            default_cols = ["candidate_id","name","email","experience_text","age","final_priority","predicted_level","predicted_fit","rule_score","reasons"]
            chosen_cols = st.multiselect("เลือกคอลัมน์ที่ต้องการแสดง", options=list(out.columns),
                                         default=[c for c in default_cols if c in out.columns])
            if chosen_cols: out = out[chosen_cols]

            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("ดาวน์โหลดผล (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                               file_name="screened_pass_fail.csv", use_container_width=True)

            st.subheader("แจ้งผลผู้สมัครทางอีเมล")
            if enable_email:
                email_col, name_col = "email", "name"
                max_reasons = st.slider("จำกัดจำนวนเหตุผลที่แสดง (เฉพาะกรณีไม่ผ่าน)", 1, 10, 3)
                preview_n = st.number_input("ดูตัวอย่างกี่รายการ", min_value=1, max_value=max(1, len(out)), value=min(3, len(out)))

                def _mk_message(rowx):
                    ctx = {
                        "name": str(rowx.get(name_col, "ผู้สมัคร")).strip() or "ผู้สมัคร",
                        "candidate_id": rowx.get("candidate_id",""),
                        "department": department,
                        "score": f"{rowx.get('final_priority',0):.2f}",
                        "reasons": "; ".join(str(rowx.get("reasons","")).split("; ")[:max_reasons])
                    }
                    passed = (rowx.get("predicted_level") == "ผ่าน")
                    subject = subj_pass if passed else subj_fail
                    body_tpl = msg_tpl_pass if passed else msg_tpl_fail
                    return _render_template(subject, ctx), _render_template(body_tpl, ctx)

                st.markdown("ตัวอย่างอีเมล")
                for i in range(int(preview_n)):
                    r = out.iloc[i]
                    addr = clean_email(r.get(email_col, ""))
                    subj, body = _mk_message(r)
                    st.code(f"To: {addr}\nSubject: {subj}\n\n{body}")

                out["email_valid"] = out[email_col].apply(is_valid_email)
                bad_rows = out[~out["email_valid"]]
                if len(bad_rows) > 0:
                    st.error(f"พบอีเมลไม่พร้อมส่ง {len(bad_rows)} รายการ (โดเมนตัวอย่าง/รูปแบบผิด/ว่าง) — จะถูกข้าม")
                    st.dataframe(bad_rows[["candidate_id","name","email"]], use_container_width=True, hide_index=True)

                test_idx = st.selectbox("ทดสอบส่งเฉพาะราย (ลำดับในตาราง):", list(range(len(out))) or [0])
                if st.button("ส่งทดสอบเฉพาะรายนี้"):
                    r = out.iloc[int(test_idx)]
                    to_email = clean_email(r.get(email_col, ""))
                    subj, body = _mk_message(r)
                    ok, info = _send_email(st.session_state.get("smtp_host", smtp_host), st.session_state.get("smtp_port", smtp_port),
                                           st.session_state.get("use_ssl", use_ssl), smtp_user, smtp_pass,
                                           from_name, from_email or smtp_user, to_email, subj, body)
                    if ok: st.success(f"ส่งสำเร็จไปที่ {to_email}")
                    else: st.error(f"ส่งไม่สำเร็จ: refused {info}")

                do_send = st.button("ส่งอีเมลแจ้งผลทั้งหมด", type="primary")
                if do_send:
                    sent, failed, skipped = 0, 0, 0
                    logs = []
                    for _, r in out.iterrows():
                        to_email = clean_email(r.get(email_col, ""))
                        if not is_valid_email(to_email):
                            skipped += 1
                            logs.append({"candidate_id": r["candidate_id"], "to": to_email, "status": "SKIPPED", "reason": "invalid/demo domain"})
                            continue
                        subj, body = _mk_message(r)
                        ok, info = _send_email(smtp_host, smtp_port, use_ssl, smtp_user, smtp_pass,
                                               from_name, from_email or smtp_user, to_email, subj, body)
                        if ok:
                            sent += 1
                            logs.append({"candidate_id": r["candidate_id"], "to": to_email, "status": "SENT", "reason": ""})
                        else:
                            failed += 1
                            logs.append({"candidate_id": r["candidate_id"], "to": to_email, "status": "FAILED", "reason": f"refused: {info}"})
                        time.sleep(1.0)
                    st.success(f"ส่งอีเมลสำเร็จ {sent} รายการ, ล้มเหลว {failed} รายการ, ข้าม {skipped} รายการ")
                    st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
            else:
                st.info("เปิดสวิตช์ 'เปิดใช้งานแจ้งผลผู้สมัครทางอีเมล' ในแถบด้านซ้ายเพื่อใช้งานฟังก์ชันนี้")

