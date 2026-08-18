import base64, csv, io, os
from pathlib import Path
import numpy as np
import streamlit as st
import cv2
import gdown
import matplotlib.pyplot as plt
import tensorflow as tf
from mtcnn import MTCNN
from PIL import Image

st.set_page_config(page_title="FaceAge · AI Age Prediction", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")

def html(s): st.markdown(s, unsafe_allow_html=True)

html('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#030914;--panel:#091022;--line:#303866;--text:#f7f5ff;--muted:#a8afc6;--pink:#e33acb;--blue:#268cff;--green:#56dc94}*{box-sizing:border-box;font-family:Inter,system-ui,sans-serif}html,body,[data-testid="stAppViewContainer"]{background:#030914;color:var(--text)}[data-testid="stAppViewContainer"]{background-image:radial-gradient(circle at 15% 15%,#0d245055,transparent 27%),radial-gradient(circle at 88% 80%,#40106644,transparent 30%)}.main .block-container{max-width:1240px;padding:0 2.1rem 3rem}header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important}.topline{height:2px;background:linear-gradient(90deg,transparent,#298cff,#e53fc7,transparent)}.nav{height:72px;display:flex;align-items:center;border-bottom:1px solid #53608044;margin-bottom:0}.brand{display:flex;align-items:center;gap:10px;font-weight:800}.brand-icon{width:32px;height:32px;border:1px solid #bd47ec;border-radius:10px;display:grid;place-items:center;color:#53c5ff;box-shadow:0 0 18px #b936df33}.note{margin-left:auto;color:#77809d;font-size:.7rem}.hero{min-height:510px;display:flex;align-items:center}.eyebrow{color:#ac8aff;font-weight:700;font-size:.75rem;letter-spacing:.14em;text-transform:uppercase;margin-bottom:14px}.hero h1{font-size:clamp(2.6rem,5vw,4.8rem);line-height:1.04;letter-spacing:-.055em;margin:0 0 20px}.gradient{background:linear-gradient(90deg,#ed43ca,#844af1,#31a5ff);-webkit-background-clip:text;color:transparent}.lead{color:var(--muted);font-size:.95rem;line-height:1.75;max-width:500px}.trust{display:grid;gap:14px;margin-top:28px;color:#c2c7da;font-size:.8rem}.trust span{display:inline-grid;place-items:center;width:24px;height:24px;border:1px solid #854ddb;border-radius:50%;color:#65d6ff;margin-right:10px}.upload-card{position:relative;border:1px solid #8741d0;border-radius:19px;padding:28px;background:linear-gradient(145deg,#11122aee,#050e1eee);box-shadow:0 0 55px #6d2bc42c}.upload-card:before,.upload-card:after{content:"";position:absolute;width:23px;height:23px}.upload-card:before{left:13px;top:13px;border-left:2px solid #2da8ff;border-top:2px solid #2da8ff}.upload-card:after{right:13px;bottom:13px;border-right:2px solid #2da8ff;border-bottom:2px solid #2da8ff}.cloud{text-align:center;font-size:3.5rem;color:#e16ae9;filter:drop-shadow(0 0 13px #9d3ad9);margin:25px 0 12px}.upload-title{text-align:center;font-weight:700}.upload-copy,.formats,.privacy{text-align:center;color:var(--muted);font-size:.72rem;line-height:1.7;margin-top:9px}.formats{color:#747c99;font-size:.65rem}.privacy{font-size:.65rem}[data-testid="stFileUploader"]{border:0;margin-top:7px}[data-testid="stFileUploaderDropzone"]{border:0!important;background:transparent!important;padding:.4rem 0!important;min-height:auto!important}[data-testid="stFileUploaderDropzoneInstructions"]{display:none}[data-testid="stFileUploaderDropzone"] button{width:100%;height:47px;border:1px solid #b539db!important;background:linear-gradient(90deg,#c832c3,#5f2ee8)!important;color:white!important;font-weight:700!important}.section{padding:64px 0}.section-head{text-align:center;margin-bottom:32px}.section-head h2{font-size:2rem;margin:0 0 10px}.section-head p{color:var(--muted);font-size:.82rem}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card,.panel{border:1px solid var(--line);border-radius:14px;background:linear-gradient(145deg,#0d1327f2,#060c1af2);padding:25px}.card{text-align:center;min-height:185px}.card:hover{border-color:#9847d9}.icon{font-size:2rem;color:#df47d2;margin-bottom:16px}.card h3{font-size:.88rem}.card p{color:var(--muted);font-size:.72rem;line-height:1.6}.results{padding:40px 0}.result-grid{display:grid;grid-template-columns:1.08fr .95fr .88fr;gap:22px}.photo{border:1px solid #8d43d7;border-radius:13px;overflow:hidden}.photo img{width:100%;height:390px;object-fit:cover;display:block}.kicker{font-size:.78rem;font-weight:600}.age{font-size:5rem;font-weight:800;line-height:1.2;background:linear-gradient(150deg,#ed4aca,#7d39ef);-webkit-background-clip:text;color:transparent}.years{font-size:.75rem}.row{display:flex;justify-content:space-between;color:#c5c9dc;font-size:.69rem;margin:19px 0 8px}.meter{height:8px;background:#151b36;border-radius:9px;overflow:hidden}.meter span{display:block;height:100%;background:linear-gradient(90deg,#d437ce,#3178ff);border-radius:9px}.metric{border:1px solid #2b376c;border-radius:11px;padding:14px;margin-top:15px;font-size:.7rem;color:#cbd0df}.metric strong{display:block;color:white;font-size:.95rem;margin-top:7px}.good{color:var(--green)!important}.explain{display:grid;gap:15px;margin:18px 0}.explain div{font-size:.7rem;color:#c4c8d9}.explain span{color:#70a6ff;border:1px solid #5b55a5;border-radius:50%;padding:1px 4px;margin-right:8px}.detail{display:grid;grid-template-columns:1fr 1fr;gap:26px}.heat{border:1px solid #8245cc;border-radius:13px;overflow:hidden}.heat img{width:100%;max-height:500px;object-fit:cover;display:block}.bar{margin:18px 0}.info{border:1px solid #2b3766;border-radius:11px;background:#0b1225;padding:15px;color:#aeb5ce;font-size:.7rem;line-height:1.6;margin-top:22px}.about{max-width:720px;margin:auto;text-align:center;color:#abb2ca;line-height:1.8;font-size:.86rem}.footer-line{border-top:1px solid #202a4d;margin-top:38px;padding:24px;text-align:center;color:#77809e;font-size:.68rem}.stButton>button{border:1px solid #384270;border-radius:9px;background:#0b1123;color:#f3f1fa;font-weight:600}.stButton>button:hover{border-color:#c03ed3;color:white}.stButton>button[kind="primary"]{border-color:#bb38d6;background:linear-gradient(90deg,#bc31bd,#562bea)}
/* Landing-page alignment and uploader containment */
.main .block-container,[data-testid="stMainBlockContainer"]{max-width:1500px!important;width:92vw!important;padding:1rem 1.25rem 2rem!important;margin:0 auto!important}
.nav{height:58px}.hero{min-height:420px}.hero h1{font-size:clamp(2.6rem,4.3vw,4.25rem);margin-bottom:16px}.trust{margin-top:20px;gap:10px}
.nav{gap:32px}.navlinks{display:flex;align-items:center;gap:30px;margin-left:36px}.navlinks.floating{position:relative;z-index:10;top:-49px;left:145px;width:max-content;height:0;margin:0}.navlinks a{color:#c9cee0;text-decoration:none;font-size:.78rem;font-weight:600;padding:9px 2px;border-bottom:2px solid transparent}.navlinks a:hover,.navlinks a.active{color:#fff;border-bottom-color:#c83dd0}.navlinks a.primary-link{padding:9px 16px;border:1px solid #3d4775;border-radius:9px}.navlinks a.primary-link:hover{border-color:#b63ed0;background:#17132d}
.upload-card{min-height:330px;padding:22px 28px 94px}.cloud{font-size:3.15rem;margin:17px 0 8px}.upload-copy,.formats{margin-top:6px}
[data-testid="stFileUploader"]{position:relative;z-index:5;width:100%;margin:-82px auto 0!important;padding:0 28px}
[data-testid="stFileUploaderDropzone"]{display:flex!important;justify-content:center!important;align-items:center!important;width:100%!important}
[data-testid="stFileUploaderDropzone"] button{width:220px!important;min-width:220px!important;margin:0 auto!important}
.privacy{position:relative;z-index:5;width:100%;text-align:center!important;margin:-19px auto 0!important;color:#9ba6c3!important}.section{padding:40px 0}.section-head{margin-bottom:26px}.footer-line{margin-top:25px;padding:18px}
/* Reference landing composition */
.nav{display:none!important}.app-navbar{height:72px;display:flex;align-items:center;width:100%;padding:0 32px;border:1px solid #17213e;border-top:0;border-radius:0 0 12px 12px;background:#050b18dd}.app-navbar .navlinks{margin-left:42px}.app-navbar .note{margin-left:auto;white-space:nowrap}
.nav{padding:0 32px;border:1px solid #17213e;border-top:0;border-radius:0 0 12px 12px;background:#050b18aa}.brand{font-size:1.1rem}.brand-icon{width:42px;height:42px;font-size:1.2rem}.navlinks.floating{left:185px}.hero{min-height:500px}.hero .eyebrow{display:inline-block;padding:7px 12px;border-radius:999px;background:linear-gradient(90deg,#351747,#09274c);font-size:.65rem}.hero h1{font-size:clamp(2.8rem,4vw,4.15rem)}.lead{font-size:.84rem;max-width:470px}.trust{gap:15px}.trust>div{display:grid;grid-template-columns:44px 1fr;align-items:center}.trust span{grid-row:1/3;width:40px;height:40px;margin-right:14px;background:#10132d}.trust b{font-size:.78rem}.trust small{color:#9fa7be;font-size:.7rem;margin-top:2px}.upload-card{min-height:455px;border-style:dashed;border-color:#cc3ed7 #29a7ff #29a7ff #cc3ed7;padding-top:38px;background:radial-gradient(circle at 50% 35%,#15204a66,transparent 40%),linear-gradient(145deg,#0d1125f5,#050d1cf5)}.cloud{width:106px;height:106px;border:1px solid #a83cd5;border-right-color:#2ca7ff;border-radius:50%;display:grid;place-items:center;margin:0 auto 15px;font-size:3rem}.upload-title{font-size:1.15rem}.upload-copy{font-size:.8rem}.formats{font-size:.7rem}[data-testid="stFileUploader"]{margin:-168px auto 0!important;max-width:330px;padding:0}[data-testid="stFileUploaderDropzone"] button{width:274px!important;min-width:274px!important;height:60px!important;font-size:1rem!important}.privacy{margin:14px auto 0!important;font-size:.75rem!important}.home-features{padding:20px 0 15px}.home-features .section-head{margin-bottom:28px}.home-features .cards{grid-template-columns:repeat(3,1fr)}.home-features .card{display:grid;grid-template-columns:90px 1fr;text-align:left;align-items:center;min-height:145px;padding:22px}.home-features .icon{grid-row:1/3;width:72px;height:72px;border:1px solid #a63ddb;border-right-color:#278eff;border-radius:50%;display:grid;place-items:center;margin:0;font-size:1.8rem}.home-features .card h3{margin:0 0 6px;font-size:.85rem}.home-features .card p{margin:0}
/* Home hero alignment */
.hero{min-height:auto!important;padding-top:24px!important;align-items:flex-start!important}.hero .eyebrow{margin:0 0 18px!important}.hero h1{margin-top:0!important;margin-bottom:18px!important}.hero .lead{margin:0 0 22px!important}.trust{display:flex!important;flex-direction:column!important;gap:14px!important;width:100%!important;margin-top:0!important}.trust>div{display:flex!important;align-items:center!important;gap:16px!important;width:100%!important;min-height:44px!important}.trust>div>span{display:flex!important;align-items:center!important;justify-content:center!important;flex:0 0 44px!important;width:44px!important;min-width:44px!important;height:44px!important;margin:0!important;line-height:1!important}.st-key-upload_panel{margin-top:64px!important}
/* Real Streamlit upload container: no overlay or negative positioning */
.st-key-upload_panel{height:430px;border:1px dashed #8e4ee8;border-right-color:#28a8ff;border-radius:18px;background:radial-gradient(circle at 50% 35%,#15204a66,transparent 42%),linear-gradient(145deg,#0d1125f5,#050d1cf5);box-shadow:0 18px 55px #05091580;padding:38px 40px 28px!important;overflow:hidden}.st-key-upload_panel .upload-content{text-align:center}.st-key-upload_panel [data-testid="stFileUploader"]{width:220px!important;max-width:220px!important;margin:26px auto 0!important;padding:0!important}.st-key-upload_panel [data-testid="stFileUploaderDropzone"]{width:220px!important;display:flex!important;justify-content:center!important}.st-key-upload_panel [data-testid="stFileUploaderDropzone"] button{width:210px!important;min-width:210px!important;height:50px!important;margin:auto!important}.st-key-upload_panel .privacy{position:static!important;width:100%!important;text-align:center!important;margin:14px auto 0!important;white-space:nowrap}.st-key-upload_panel .cloud{width:94px;height:94px}.st-key-upload_panel .upload-title{margin-top:4px}
/* Full-height error state with an in-flow bottom footer */
[data-testid="stMainBlockContainer"]:has(.no-face-page)>div[data-testid="stVerticalBlock"]{min-height:calc(100vh - 2rem)!important;display:flex!important;flex-direction:column!important}.st-key-error_main_content{flex:1 1 auto!important;width:100%!important;display:flex!important;flex-direction:column!important;justify-content:center!important}.st-key-error_main_content>div[data-testid="stVerticalBlock"]{width:100%!important}.no-face-page .section{padding-top:20px!important;padding-bottom:0!important}[data-testid="stMainBlockContainer"]:has(.no-face-page) div[data-testid="stElementContainer"]:has(.footer-line){margin-top:auto!important;width:100%!important}.footer-line{width:100%}
/* Results-only centered primary action */
.st-key-result_action{width:100%!important;display:block!important;margin-top:24px!important}.st-key-result_action>div[data-testid="stVerticalBlock"],.st-key-result_action div[data-testid="stVerticalBlock"]{width:100%!important;display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important}.st-key-result_action div[data-testid="stElementContainer"]{width:100%!important;display:flex!important;align-items:center!important;justify-content:center!important}.st-key-result_action [data-testid="stButton"]{width:100%!important;display:flex!important;align-items:center!important;justify-content:center!important}.st-key-result_action [data-testid="stButton"]>button{flex:0 0 240px!important;width:240px!important;min-width:220px!important;max-width:260px!important;height:48px!important;min-height:48px!important;max-height:48px!important;padding:0 18px!important;border-radius:10px!important}.st-key-result_action [data-testid="stButton"]>button:hover{transform:translateY(-1px);box-shadow:0 0 16px #b83ed755!important}
/* Results dashboard */
.st-key-analyzed_panel,.st-key-prediction_panel{height:100%!important;min-height:100%;border:1px solid #303a69;border-radius:16px;background:linear-gradient(145deg,#0d1428f5,#060c1bf5);padding:22px!important;box-shadow:0 15px 40px #02061166}.st-key-analyzed_panel .result-card,.st-key-prediction_panel .result-card{height:auto!important;border:0!important;border-radius:0!important;background:transparent!important;padding:0!important;box-shadow:none!important}div[data-testid="stHorizontalBlock"]:has(.st-key-analyzed_panel):has(.st-key-prediction_panel){align-items:stretch!important}div[data-testid="stHorizontalBlock"]:has(.st-key-analyzed_panel):has(.st-key-prediction_panel)>div[data-testid="stColumn"]{display:flex!important;align-items:stretch!important}div[data-testid="stHorizontalBlock"]:has(.st-key-analyzed_panel):has(.st-key-prediction_panel)>div[data-testid="stColumn"]>div{width:100%!important;height:100%!important}.st-key-analyzed_panel>div[data-testid="stVerticalBlock"],.st-key-prediction_panel>div[data-testid="stVerticalBlock"]{height:100%!important}.st-key-analyzed_panel .result-photo{flex:1 1 auto}.st-key-analyzed_panel .result-photo img{object-fit:contain!important}
div[data-testid="stHorizontalBlock"]:has(.st-key-analyzed_panel):has(.st-key-prediction_panel)>div[data-testid="stColumn"]>div[data-testid="stVerticalBlock"]{height:100%!important;align-self:stretch!important}div[data-testid="stElementContainer"]:has(>.st-key-analyzed_panel),div[data-testid="stElementContainer"]:has(>.st-key-prediction_panel){height:100%!important;display:flex!important;align-items:stretch!important}.st-key-analyzed_panel,.st-key-prediction_panel{align-self:stretch!important}.st-key-analyzed_panel>div[data-testid="stVerticalBlock"],.st-key-prediction_panel>div[data-testid="stVerticalBlock"]{display:flex!important;flex-direction:column!important;min-height:100%!important}.st-key-analyzed_panel .result-card{height:100%!important;min-height:100%!important;display:flex!important;flex-direction:column!important}.st-key-analyzed_panel .status-line{margin-top:auto!important;padding-top:14px!important}
.st-key-result_action .stButton,.st-key-retry_action .stButton{display:flex!important;justify-content:center!important;width:100%!important}.st-key-result_action .stButton>button,.st-key-retry_action .stButton>button{width:200px!important;min-width:180px!important;max-width:220px!important;height:46px!important;min-height:46px!important;padding:0 18px!important;border-radius:10px!important;margin:0 auto!important}.st-key-result_action .stButton>button:hover,.st-key-retry_action .stButton>button:hover{transform:translateY(-1px);box-shadow:0 0 16px #b83ed755!important}
.st-key-retry_button_container{width:100%!important;display:block!important;margin:32px auto 0!important;padding:0!important;float:none!important;position:static!important}.st-key-retry_button_container>div[data-testid="stVerticalBlock"],.st-key-retry_button_container div[data-testid="stVerticalBlock"]{width:100%!important;display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important}.st-key-retry_button_container div[data-testid="stElementContainer"]{width:100%!important;display:flex!important;justify-content:center!important;align-items:center!important}.st-key-retry_button_container [data-testid="stButton"]{width:100%!important;display:flex!important;justify-content:center!important;align-items:center!important;margin:0 auto!important;float:none!important;position:static!important}.st-key-retry_button_container [data-testid="stButton"]>button{flex:0 0 230px!important;width:230px!important;min-width:230px!important;max-width:230px!important;height:50px!important;min-height:50px!important;max-height:50px!important;padding:0 18px!important;margin-left:auto!important;margin-right:auto!important;border-radius:10px!important;float:none!important;position:static!important}.st-key-retry_button_container [data-testid="stButton"]>button:hover{transform:translateY(-1px);box-shadow:0 0 16px #b83ed755!important}
.age-group-box{margin-top:17px;padding:14px;border:1px solid #7045bb;border-radius:11px;background:linear-gradient(120deg,#17112e,#0a1730)}.age-group-box span{display:block;color:#aab2ca;font-size:.68rem}.age-group-box strong{display:block;margin:6px 0 3px;font-size:1.15rem;color:#f1eaff}.age-group-box small{color:#718fc7;font-size:.67rem}
.performance-section{padding:42px 0 10px}.performance-head{text-align:center;margin-bottom:22px}.performance-head .eyebrow{margin-bottom:8px}.performance-head h2{margin:0 0 7px;font-size:1.65rem}.performance-head p{margin:0;color:#929cb7;font-size:.72rem}.metric-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.performance-card{min-height:148px;border:1px solid #303a69;border-radius:13px;background:linear-gradient(145deg,#0d1428f2,#070d1cf2);padding:18px;transition:transform .2s,border-color .2s}.performance-card:hover{transform:translateY(-2px);border-color:#7449c5}.performance-kicker{color:#a8b1c9;font-size:.65rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}.performance-value{margin:15px 0 9px;font-size:1.65rem;font-weight:800;background:linear-gradient(90deg,#e34bcf,#4b91ff);-webkit-background-clip:text;color:transparent}.performance-label{color:#d4d8e7;font-size:.72rem;font-weight:600}.performance-help{margin-top:5px;color:#828ca8;font-size:.65rem}.performance-note{text-align:center;margin:14px 0 0;color:#7f89a5;font-size:.64rem}
.results-header{text-align:center;padding:34px 0 24px}.results-header .eyebrow{display:inline-block;padding:7px 13px;border:1px solid #7544bd;border-radius:999px;background:#24133f;color:#d595ff}.results-header h1{font-size:2.1rem;margin:13px 0 8px}.results-header p{color:#9fa8c0;font-size:.78rem}.result-title{font-size:.82rem;font-weight:700;margin-bottom:14px}.result-card{height:100%;border:1px solid #303a69;border-radius:16px;background:linear-gradient(145deg,#0d1428f5,#060c1bf5);padding:22px;box-shadow:0 15px 40px #02061166}.detected-label{display:inline-block;margin-bottom:10px;padding:5px 9px;border-radius:6px;background:#17213a;color:#aeb8d4;font-size:.65rem}.result-photo{width:100%;min-height:260px;display:flex;align-items:center;justify-content:center;border:1px solid #7546c6;border-radius:12px;background:#050a15;padding:10px;box-shadow:0 0 24px #5d39b426}.result-photo img{display:block;width:auto;max-width:100%;height:auto;max-height:520px;object-fit:contain;margin:0 auto;border-radius:9px}.status-line{display:flex;gap:22px;margin-top:14px;color:#9ce6ba;font-size:.7rem}.prediction-age{font-size:4.4rem;font-weight:800;line-height:1;background:linear-gradient(135deg,#ed4aca,#477cff);-webkit-background-clip:text;color:transparent}.prediction-age small{font-size:.8rem;color:#d7dbeb;-webkit-text-fill-color:#d7dbeb}.uncertainty-box{margin:18px 0;padding:14px;border:1px solid #2f3c70;border-radius:11px;background:#0a1225}.uncertainty-box span{display:block;color:#9da7c1;font-size:.68rem}.uncertainty-box strong{display:block;margin:7px 0 4px;font-size:1.05rem}.chart-heading{margin:16px 0 3px;font-size:.82rem;font-weight:700}.explain-section{padding:50px 0 18px}.cam-card{border:1px solid #303a69;border-radius:14px;background:#091022;padding:15px;height:100%}.cam-card img{width:100%;aspect-ratio:1/1;object-fit:contain;border-radius:9px;background:#030914}.cam-label{text-align:center;font-size:.76rem;font-weight:700;margin-bottom:11px}.attention-legend{text-align:center;color:#9ea7bf;font-size:.68rem;margin-top:9px}.reasoning-card{margin-top:22px;border:1px solid #6242a8;border-radius:14px;padding:20px;background:linear-gradient(120deg,#17112c,#09162c);color:#adb6ce;font-size:.76rem;line-height:1.7}.reasoning-card b{display:block;color:#fff;font-size:.88rem;margin-bottom:6px}.signal-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:18px 0 26px}.signal-card{text-align:center;border:1px solid #2f3968;border-radius:11px;background:#0a1123;padding:17px 10px;font-size:.72rem;color:#d2d6e6}
/* Selected-file remove control: intentionally separate from the Upload button */
.st-key-upload_panel [data-testid="stFileUploaderDropzone"] button:not([data-testid="stFileUploaderDeleteBtn"]){width:210px!important;min-width:210px!important;max-width:210px!important;height:50px!important}
.st-key-upload_panel [data-testid="stFileUploaderFile"]{width:100%!important;display:flex!important;align-items:center!important;background:#0a1122!important;border:1px solid #27345f!important;border-radius:10px!important;padding:7px 9px!important}
.st-key-upload_panel [data-testid="stFileUploaderDeleteBtn"]{flex:0 0 36px!important;width:36px!important;min-width:36px!important;max-width:36px!important;height:36px!important;min-height:36px!important;max-height:36px!important;padding:0!important;margin:0 0 0 auto!important;border:1px solid #8e48ce!important;border-radius:50%!important;background:#11152a!important;box-shadow:none!important;display:inline-flex!important;align-items:center!important;justify-content:center!important;color:#eee9ff!important}
.st-key-upload_panel [data-testid="stFileUploaderDeleteBtn"]:hover{border-color:#da4ee1!important;background:#201338!important;box-shadow:0 0 12px #b83ed766!important}
.st-key-upload_panel [data-testid="stFileUploaderDeleteBtn"] svg{width:17px!important;height:17px!important;margin:0!important}
/* Streamlit 1.x current DOM: test ID is on the file-chip delete wrapper. */
.st-key-upload_panel [data-testid="stFileChipDeleteBtn"]{flex:0 0 34px!important;flex-grow:0!important;flex-shrink:0!important;width:34px!important;min-width:34px!important;max-width:34px!important;height:34px!important;min-height:34px!important;max-height:34px!important;margin:0 0 0 auto!important;padding:0!important;display:flex!important;align-items:center!important;justify-content:center!important}
.st-key-upload_panel [data-testid="stFileChipDeleteBtn"]>button{flex:0 0 34px!important;flex-grow:0!important;flex-shrink:0!important;width:34px!important;min-width:34px!important;max-width:34px!important;height:34px!important;min-height:34px!important;max-height:34px!important;margin:0!important;padding:0!important;border:1px solid #8e48ce!important;border-radius:50%!important;background:#11152a!important;background-image:none!important;box-shadow:none!important;display:inline-flex!important;align-items:center!important;justify-content:center!important;color:#eee9ff!important}
.st-key-upload_panel [data-testid="stFileChipDeleteBtn"]>button:hover{border-color:#da4ee1!important;background:#201338!important;box-shadow:0 0 12px #b83ed766!important}
.st-key-upload_panel [data-testid="stFileChipDeleteBtn"]>button svg{width:17px!important;height:17px!important;margin:0!important}
/* Verified in this Streamlit bundle: aria-label is "Remove <filename>" or
   "Cancel upload of <filename>" on the actual inner BaseButton. */
.st-key-upload_panel [data-testid="stFileUploader"] [data-testid="stFileChipDeleteBtn"]{box-sizing:border-box!important;flex:0 0 32px!important;flex-grow:0!important;flex-shrink:0!important;width:32px!important;min-width:32px!important;max-width:32px!important;height:32px!important;min-height:32px!important;max-height:32px!important;padding:0!important;margin:0 0 0 auto!important;background:transparent!important;border:0!important}
.st-key-upload_panel [data-testid="stFileUploader"] button[aria-label^="Remove "],.st-key-upload_panel [data-testid="stFileUploader"] button[aria-label^="Cancel upload of "]{box-sizing:border-box!important;flex:0 0 32px!important;flex-grow:0!important;flex-shrink:0!important;align-self:center!important;width:32px!important;min-width:32px!important;max-width:32px!important;height:32px!important;min-height:32px!important;max-height:32px!important;padding:0!important;margin:0!important;border:1px solid #8e48ce!important;border-radius:50%!important;background:#11152a!important;background-image:none!important;box-shadow:none!important;display:inline-flex!important;align-items:center!important;justify-content:center!important;overflow:hidden!important}
.st-key-upload_panel [data-testid="stFileUploader"] button[aria-label^="Remove "]:hover,.st-key-upload_panel [data-testid="stFileUploader"] button[aria-label^="Cancel upload of "]:hover{background:#201338!important;background-image:none!important;border-color:#da4ee1!important;box-shadow:0 0 10px #b83ed755!important}
@media(max-width:1050px){.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:800px){.main .block-container,[data-testid="stMainBlockContainer"]{width:100%!important;padding:.7rem 1rem 1.5rem!important}.nav{height:58px}.navlinks.floating{top:-4px;left:0;height:42px;width:100%;gap:22px}.app-navbar{height:auto;min-height:68px;padding:12px 16px;gap:14px;flex-wrap:wrap}.app-navbar .navlinks{order:3;width:100%;margin:0;justify-content:center}.note{display:none}.hero{min-height:auto;padding:25px 0 18px}.hero h1{font-size:2.65rem}.upload-card{min-height:320px}.cards,.result-grid,.detail,.home-features .cards{grid-template-columns:1fr}.photo img{height:auto}.section{padding:30px 0}[data-testid="stFileUploader"]{margin:-82px auto 0!important;padding:0 22px}.st-key-upload_panel{height:410px;padding:30px 18px 22px!important}.st-key-upload_panel [data-testid="stFileUploader"]{margin:24px auto 0!important;padding:0!important}.st-key-upload_panel .privacy{white-space:normal}.result-photo{height:auto;min-height:220px}.signal-grid{grid-template-columns:1fr 1fr}}
@media(max-width:540px){.metric-grid{grid-template-columns:1fr}}
@media(max-width:800px){.st-key-analyzed_panel,.st-key-prediction_panel{height:auto!important;min-height:0!important;align-self:auto!important}.st-key-analyzed_panel>div[data-testid="stVerticalBlock"],.st-key-prediction_panel>div[data-testid="stVerticalBlock"],.st-key-analyzed_panel .result-card{height:auto!important;min-height:0!important}div[data-testid="stHorizontalBlock"]:has(.st-key-analyzed_panel):has(.st-key-prediction_panel)>div[data-testid="stColumn"]{display:block!important}div[data-testid="stHorizontalBlock"]:has(.st-key-analyzed_panel):has(.st-key-prediction_panel)>div[data-testid="stColumn"]>div{height:auto!important}}
@media(max-width:800px){.st-key-upload_panel{margin-top:20px!important}}
</style>''')

def go(page): st.session_state.page=page
st.session_state.setdefault("page", "Home")
html('<div class="topline"></div><div class="nav"><div class="brand"><span class="brand-icon">◉</span> FaceAge</div><span class="note">AI-powered age estimation · private by design</span></div>')
requested=st.query_params.get("page")
if requested in ("Home","About"):
    st.session_state.page=requested
elif requested=="New":
    for k in ("image","result","tensor","original","cam","overlay"): st.session_state.pop(k,None)
    st.session_state.page="Home"; st.query_params.clear(); st.rerun()
active=st.session_state.page
html(f'''<div class="app-navbar"><div class="brand"><span class="brand-icon">◉</span><span>FaceAge</span></div><div class="navlinks"><a class="{'active' if active=='Home' else ''}" href="?page=Home" target="_self">Home</a><a class="{'active' if active=='About' else ''}" href="?page=About" target="_self">About</a><a href="?page=New" target="_self">New Analysis</a></div><div class="note">◈ &nbsp; AI-powered age estimation • Private by design</div></div>''')

MODEL_PATH="best_soft_label_model.keras"
GDRIVE_FILE_ID="1oN5aI1HHgZNga2qZ5ADDXzQBoW0bI8U-"

def patch_mc_dropout(model_obj):
    count=0
    def patch(container):
        nonlocal count
        for layer in getattr(container,"layers",getattr(container,"submodules",[])):
            if isinstance(layer,tf.keras.layers.Dropout) and not getattr(layer,"_mc_patched",False):
                original=layer.call
                layer.call=lambda inputs,*args,_original=original,**kwargs:_original(inputs,training=True)
                layer._mc_patched=True; count+=1
            if hasattr(layer,"layers") or hasattr(layer,"submodules"): patch(layer)
    patch(model_obj); return count

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",MODEL_PATH,quiet=False)
    loaded=tf.keras.models.load_model(MODEL_PATH,compile=False)
    patch_mc_dropout(loaded)
    return loaded

@st.cache_resource(show_spinner=False)
def load_detector(): return MTCNN()

def crop_and_align_face(image):
    array=np.asarray(image.convert("RGB")); faces=load_detector().detect_faces(array)
    if not faces: return None,None
    face=max(faces,key=lambda r:r["box"][2]*r["box"][3]); x,y,w,h=face["box"]; x,y=max(0,x),max(0,y)
    px,py=int(w*.2),int(h*.2); x1,y1=max(0,x-px),max(0,y-py); x2,y2=min(array.shape[1],x+w+px),min(array.shape[0],y+h+py)
    crop=cv2.resize(array[y1:y2,x1:x2],(256,256)).astype(np.float32)
    return np.expand_dims(crop,0),(x1,y1,x2,y2)

def predict_age(tensor,num_passes=10,use_tta=True):
    model=load_model(); bins=np.arange(101); runs=[]
    for _ in range(num_passes):
        probs=model(tensor,training=False).numpy()[0]
        if use_tta: probs=(probs+model(tensor[:,:,::-1,:],training=False).numpy()[0])/2
        runs.append((np.sum(bins*probs),probs))
    ages=[r[0] for r in runs]
    return float(np.mean(ages)),float(np.std(ages)),np.mean([r[1] for r in runs],axis=0)

def generate_gradcam(tensor):
    model=load_model(); backbone=target=None
    for layer in model.layers:
        if hasattr(layer,"get_layer"):
            try: target=layer.get_layer("top_conv"); backbone=layer; break
            except ValueError: pass
    if backbone is None:
        for layer in reversed(model.layers):
            if layer.name=="top_conv" or isinstance(layer,tf.keras.layers.Conv2D): target=layer; break
        grad_model=tf.keras.Model(model.inputs,[target.output,model.output])
        with tf.GradientTape() as tape:
            conv,preds=grad_model(tensor,training=False); bins=tf.cast(tf.range(tf.shape(preds)[-1]),preds.dtype); expected=tf.reduce_sum(preds*bins,axis=-1)
    else:
        gm=tf.keras.Model(backbone.inputs,[target.output,backbone.output]); idx=model.layers.index(backbone); head=model.layers[idx+1:]
        gap=next(l for l in head if isinstance(l,tf.keras.layers.GlobalAveragePooling2D)); gmp=next(l for l in head if isinstance(l,tf.keras.layers.GlobalMaxPooling2D)); concat=next(l for l in head if isinstance(l,tf.keras.layers.Concatenate)); dense=next(l for l in head if isinstance(l,tf.keras.layers.Dense))
        with tf.GradientTape() as tape:
            conv,bb=gm(tensor,training=False); tape.watch(conv); preds=dense(concat([gap(bb),gmp(bb)])); bins=tf.cast(tf.range(tf.shape(preds)[-1]),preds.dtype); expected=tf.reduce_sum(preds*bins,axis=-1)
    grads=tape.gradient(expected,conv); weights=tf.reduce_mean(grads,axis=(0,1,2)); heat=tf.reduce_sum(conv[0]*weights,axis=-1); heat=tf.maximum(heat,0)/(tf.reduce_max(heat)+1e-8)
    return heat.numpy()

def cam_images(tensor,heat):
    base=np.clip(tensor[0],0,255).astype(np.uint8); gray=Image.fromarray((heat*255).astype(np.uint8)).resize((256,256),Image.Resampling.BILINEAR); values=np.asarray(gray)/255
    colored=(plt.get_cmap("turbo")(values)[...,:3]*255).astype(np.uint8); overlay=np.clip(.55*base+.45*colored,0,255).astype(np.uint8)
    return Image.fromarray(base),Image.fromarray(colored),Image.fromarray(overlay)

def uri(image):
    out=io.BytesIO(); image.convert("RGB").save(out,"JPEG",quality=90)
    return "data:image/jpeg;base64,"+base64.b64encode(out.getvalue()).decode()

def get_age_group(predicted_age):
    """Map a numerical age prediction to a display category."""
    age=float(predicted_age)
    if not np.isfinite(age): return "Unavailable", "Age unavailable"
    if age<13: return "Child", "0–12 years"
    if age<20: return "Teen", "13–19 years"
    if age<36: return "Young Adult", "20–35 years"
    if age<60: return "Adult", "36–59 years"
    return "Senior", "60+ years"

def calculate_evaluation_metrics(y_true, model_output):
    """Calculate regression metrics from labels and DEX outputs/predictions."""
    labels=np.asarray(y_true,dtype=np.float64).reshape(-1)
    outputs=np.asarray(model_output,dtype=np.float64)
    if outputs.ndim==2:
        # DEX produces a probability distribution over age classes.
        age_bins=np.arange(outputs.shape[1],dtype=np.float64)
        predictions=np.sum(outputs*age_bins[None,:],axis=1)
    else:
        predictions=outputs.reshape(-1)
    if labels.size!=predictions.size:
        raise ValueError("Evaluation labels and predictions have different lengths.")
    valid=np.isfinite(labels)&np.isfinite(predictions)
    if not np.any(valid):
        raise ValueError("Evaluation data contains no valid numeric label/prediction pairs.")
    labels,predictions=labels[valid],predictions[valid]
    errors=predictions-labels
    return {
        "mae":float(np.mean(np.abs(errors))),
        "rmse":float(np.sqrt(np.mean(np.square(errors)))),
        "within_5":float(np.mean(np.abs(errors)<=5.0)*100.0),
        "samples":int(labels.size),
    }

@st.cache_data(show_spinner=False)
def evaluate_model_performance():
    """Load existing labeled evaluation predictions without retraining the model."""
    root=Path(__file__).resolve().parent
    npz_names=("evaluation_predictions.npz","test_predictions.npz","validation_predictions.npz")
    for name in npz_names:
        path=root/name
        if path.exists():
            try:
                with np.load(path,allow_pickle=False) as data:
                    label_key=next((k for k in ("y_true","labels","actual_ages") if k in data),None)
                    output_key=next((k for k in ("y_pred","predictions","probabilities","mean_probs") if k in data),None)
                    if not label_key or not output_key:
                        raise ValueError("Expected y_true/labels and y_pred/predictions/probabilities arrays.")
                    metrics=calculate_evaluation_metrics(data[label_key],data[output_key])
                return {"status":"available","source":name,**metrics}
            except Exception as exc:
                return {"status":"invalid","message":f"Invalid evaluation data: {exc}"}
    csv_names=("evaluation_predictions.csv","test_predictions.csv","validation_predictions.csv")
    for name in csv_names:
        path=root/name
        if path.exists():
            try:
                with path.open("r",encoding="utf-8-sig",newline="") as handle:
                    rows=list(csv.DictReader(handle))
                if not rows: raise ValueError("The evaluation file is empty.")
                label_col=next((k for k in ("y_true","actual_age","age","label") if k in rows[0]),None)
                pred_col=next((k for k in ("y_pred","predicted_age","prediction") if k in rows[0]),None)
                if not label_col or not pred_col:
                    raise ValueError("Expected actual-age and predicted-age columns.")
                metrics=calculate_evaluation_metrics([r[label_col] for r in rows],[r[pred_col] for r in rows])
                return {"status":"available","source":name,**metrics}
            except Exception as exc:
                return {"status":"invalid","message":f"Invalid evaluation data: {exc}"}
    return {"status":"unavailable","message":"Evaluation dataset unavailable"}


def features():
    html('''<section class="section"><div class="section-head"><div class="eyebrow">Built for clarity</div><h2>Why FaceAge?</h2><p>Prediction, uncertainty, and explainability in one workflow.</p></div><div class="cards">
    <div class="card"><div class="icon">◉</div><h3>AI-Powered Prediction</h3><p>Deep learning based facial age estimation.</p></div><div class="card"><div class="icon">◎</div><h3>MC Dropout Uncertainty</h3><p>Multiple stochastic inference passes provide an uncertainty estimate.</p></div><div class="card"><div class="icon">☷</div><h3>Test-Time Augmentation</h3><p>Prediction stability is improved using augmented inference.</p></div><div class="card"><div class="icon">▢</div><h3>Grad-CAM Explainability</h3><p>Visualize facial regions receiving stronger model attention.</p></div><div class="card"><div class="icon">ϟ</div><h3>Face Detection & Alignment</h3><p>Faces are detected and standardized before inference.</p></div></div></section>''')

def home_features():
    html('''<section class="home-features"><div class="section-head"><div class="eyebrow">Built for clarity</div><h2>Why FaceAge?</h2><p>Prediction, uncertainty, and explainability in one workflow.</p></div><div class="cards"><div class="card"><div class="icon">◉</div><div><h3>AI-Powered Prediction</h3><p>Deep learning model trained for accurate age estimation.</p></div></div><div class="card"><div class="icon">▥</div><div><h3>Explainable AI</h3><p>Understand which visual patterns influenced the prediction.</p></div></div><div class="card"><div class="icon">◆</div><div><h3>Privacy Focused</h3><p>Images are processed securely and are not stored.</p></div></div></div></section>''')

def home():
    l,r=st.columns([.84,1.16],gap="medium",vertical_alignment="top")
    with l: html('''<div class="hero"><div><div class="eyebrow">AI-powered face analysis</div><h1>Discover the<br><span class="gradient">Age Behind</span><br>the Face</h1><p class="lead">Upload a clear, front-facing image and our AI will estimate apparent age with confidence and visual explainability.</p><div class="trust"><div><span>✓</span>AI-powered prediction</div><div><span>✓</span>Confidence and age range</div><div><span>✓</span>Privacy-focused processing</div></div></div></div>''')
    with r:
        with st.container(key="upload_panel"):
            html('<div class="upload-content"><div class="cloud">⇧</div><div class="upload-title">Upload Face Image</div><div class="upload-copy">Drag and drop your image here<br>or click to browse</div><div class="formats">JPG, JPEG, PNG • Max 10 MB</div></div>')
            f=st.file_uploader("Choose file",type=["jpg","jpeg","png"],label_visibility="collapsed")
            html('<div class="privacy">▣ &nbsp; Processed securely and not stored.</div>')
        if f:
            if f.size>10*1024*1024: st.error("Please choose an image smaller than 10 MB.")
            else:
                try:
                    raw=f.getvalue(); im=Image.open(io.BytesIO(raw)); im.verify(); im=Image.open(io.BytesIO(raw)).convert("RGB")
                    stage=st.empty(); progress=st.progress(5)
                    stage.info("Analyzing Face… · Face Detection")
                    tensor,box=crop_and_align_face(im); progress.progress(25)
                    if tensor is None:
                        progress.empty(); stage.empty(); st.session_state.image=raw; go("No face"); st.rerun()
                    stage.info("Analyzing Face… · Face Alignment → Model Inference")
                    progress.progress(45); age,std,probs=predict_age(tensor,num_passes=5,use_tta=True); progress.progress(90)
                    stage.info("Analyzing Face… · Uncertainty Estimation")
                    original=Image.fromarray(np.clip(tensor[0],0,255).astype(np.uint8)); progress.progress(100)
                    def pack(img): out=io.BytesIO(); img.save(out,"PNG"); return out.getvalue()
                    st.session_state.image=raw; st.session_state.result=(age,std,probs); st.session_state.tensor=tensor; st.session_state.original=pack(original)
                    go("Results"); st.rerun()
                except Exception:
                    st.error("Analysis could not be completed. Please verify the model file and try a clear JPG or PNG.")
    home_features()

def results():
    if "image" not in st.session_state: go("Home"); st.rerun()
    im=Image.open(io.BytesIO(st.session_state.image)).convert("RGB"); age,std,probs=st.session_state.result; low,high=age-std,age+std
    age_group,age_group_range=get_age_group(age)
    html('<div class="results-header"><div class="eyebrow">✓ Analysis complete</div><h1>Your Age Prediction</h1><p>Review the predicted age, uncertainty, and model explanation.</p></div>')
    left,right=st.columns([1.04,.96],gap="medium",vertical_alignment="top")
    with left:
        with st.container(key="analyzed_panel"):
            html(f'''<div class="result-card"><div class="result-title">Analyzed Face</div><div class="detected-label">Original Uploaded Image</div><div class="result-photo"><img src="{uri(im)}" alt="Complete uploaded image"></div><div class="status-line"><span>✓ Face detected</span></div></div>''')
    with right:
        with st.container(key="prediction_panel"):
            html(f'''<div class="result-card"><div class="result-title">Predicted Age</div><div class="prediction-age">{age:.1f} <small>years</small></div><div class="age-group-box"><span>Age Group</span><strong>{age_group}</strong><small>{age_group_range}</small></div><div class="uncertainty-box"><span>Uncertainty Range (±1σ)</span><strong>{low:.1f} – {high:.1f} years</strong><span>σ = {std:.2f}</span></div><p style="color:#929cb7;font-size:.67rem;line-height:1.6;margin:0">Represents prediction variability across stochastic inference passes.</p><div class="chart-heading">Age Probability Distribution</div><p style="color:#8993ae;font-size:.66rem;margin:0">Model output across age classes</p></div>''')
            plot_probs=np.asarray(probs,dtype=np.float64).reshape(-1)
            plot_probs=np.nan_to_num(plot_probs,nan=0.0,posinf=0.0,neginf=0.0)
            plot_probs=np.clip(plot_probs,0.0,None)
            total=float(plot_probs.sum())
            if total>0: plot_probs=plot_probs/total
            fig,ax=plt.subplots(figsize=(7,3.15)); fig.patch.set_facecolor("#091022"); ax.set_facecolor("#091022"); ages=np.arange(plot_probs.size); ax.plot(ages,plot_probs,color="#519cff",lw=2.2); ax.fill_between(ages,plot_probs,color="#8a42ed",alpha=.22); ax.axvline(age,color="#f04dcc",ls="--",lw=1.5,label=f"Prediction {age:.1f}"); ax.axvspan(low,high,color="#36b8ff",alpha=.12,label="±1σ"); ax.grid(color="#7180a4",alpha=.15); ax.tick_params(colors="#a8afc6",labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel("Age Class",color="#a8afc6",fontsize=8); ax.set_ylabel("Probability",color="#a8afc6",fontsize=8); ax.legend(frameon=False,labelcolor="#bfc6da",fontsize=7,loc="upper right"); fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    
            st.session_state.cam=pack(cam); st.session_state.overlay=pack(overlay)
    original=Image.open(io.BytesIO(st.session_state.original)); cam=Image.open(io.BytesIO(st.session_state.cam)); overlay=Image.open(io.BytesIO(st.session_state.overlay))
    cam_cols=st.columns(3,gap="medium")
    for col,title,picture in zip(cam_cols,["Original Image","Grad-CAM Heatmap","Attention Overlay"],[original,cam,overlay]):
        with col: html(f'<div class="cam-card"><div class="cam-label">{title}</div><img src="{uri(picture)}" alt="{title}"><div class="attention-legend">Cool / Low Attention → Warm / High Attention</div></div>')
    html('''<div class="reasoning-card"><b>💡 How the Model Saw This Face</b>Grad-CAM highlights regions where convolutional features contributed strongly to the model output. The visualization reflects model attention across facial geometry, the eye area, skin texture, jawline, and overall structure. It does not establish biological age or a medical characteristic.</div><div class="section-head" style="margin-top:34px"><h2>Visual Signals Considered</h2><p>Qualitative visual patterns used by the learned feature representation.</p></div><div class="signal-grid"><div class="signal-card">Facial Geometry</div><div class="signal-card">Skin Texture</div><div class="signal-card">Eye Area</div><div class="signal-card">Fine Lines & Contrast</div><div class="signal-card">Overall Structure</div></div>''')
    with st.container(key="result_action"):
        if st.button("↻  Upload Another Image",type="primary",use_container_width=False):
            for k in ("image","result","tensor","original","cam","overlay"): st.session_state.pop(k,None)
            go("Home"); st.rerun()

def details():
    if "image" not in st.session_state: go("Home"); st.rerun()
    if "cam" not in st.session_state:
        with st.spinner("Generating Grad-CAM attention maps…"):
            heat=generate_gradcam(st.session_state.tensor); original,cam,overlay=cam_images(st.session_state.tensor,heat)
            def pack(img): out=io.BytesIO(); img.save(out,"PNG"); return out.getvalue()
            st.session_state.cam=pack(cam); st.session_state.overlay=pack(overlay)
    original=Image.open(io.BytesIO(st.session_state.original)); cam=Image.open(io.BytesIO(st.session_state.cam)); overlay=Image.open(io.BytesIO(st.session_state.overlay))
    html('<section class="results"><div class="eyebrow">Model Explainability</div><h2>See which facial regions influenced the model’s prediction</h2><p style="color:#9fa8c0;font-size:.76rem;margin-bottom:26px">Warm regions indicate stronger model attention; cool regions indicate lower attention.</p></section>')
    cols=st.columns(3,gap="medium")
    for col,title,img in zip(cols,["Original Image","Grad-CAM Heatmap","Attention Overlay"],[original,cam,overlay]):
        with col: html(f'<div class="kicker">{title}</div>'); st.image(img,use_container_width=True)
    html('<div class="info" style="text-align:center">Low Attention &nbsp; <span style="color:#36a8ff">●</span> ───────── <span style="color:#ff4c47">●</span> &nbsp; High Attention</div><div class="section-head" style="margin-top:38px"><h2>Visual Signals Considered</h2><p>Qualitative model-attention categories; no unsupported percentages are inferred.</p></div>')
    names=["Eye Area","Skin Texture","Forehead","Cheek Area","Jawline","Overall Facial Structure"]
    html('<div class="cards">'+''.join(f'<div class="card" style="min-height:110px"><h3>{n}</h3><p>Model attention visualized in the Grad-CAM view</p></div>' for n in names)+'</div><div class="info"><b>How the AI interpreted the image</b><br><br>For this prediction, the model placed attention on facial regions and visual patterns that contributed to the estimated age distribution. The heatmap shows model attention, not biological age or medical characteristics.</div>')
    if st.button("← Back to results"): go("Results"); st.rerun()

p=st.session_state.page
if p=="Home" or p=="New analysis": home()
elif p=="Results": results()
elif p=="Details": details()
elif p=="No face":
    with st.container(key="error_main_content"):
        html('<div class="no-face-page"><section class="section"><div class="section-head"><div class="icon">◎</div><h2>Face not detected</h2><p>Please upload a clearer image with a visible face.</p></div></section></div>')
        with st.container(key="retry_button_container"):
            if st.button("↻  Try Another Image",type="primary",use_container_width=False):
                for k in ("image","result","tensor","original","cam","overlay"): st.session_state.pop(k,None)
                go("Home"); st.rerun()
else:
    html('''<section class="section"><div class="section-head"><div class="eyebrow">About the project</div><h2>AI age estimation, made understandable</h2></div><div class="about">FaceAge is a responsive demonstration for apparent-age estimation. It pairs image analysis with confidence reporting and visual explainability, making the output more useful than a single number.<br><br>The repository exposes a stable <code>predict</code> function where your trained TensorFlow model can be connected. Results are informational and must not be used for medical, legal, identity, eligibility, or other high-impact decisions.</div></section>'''); features()
html('<div class="footer-line">FaceAge · AI-powered age prediction · Responsive and privacy focused</div>')
