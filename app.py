import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io
import altair as alt
import requests
from fpdf import FPDF
import json
import os
from scipy.optimize import fsolve
from streamlit_authenticator import Authenticate
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import validators
from google.cloud import storage
import boto3
from botocore.exceptions import ClientError
import uuid
import urllib.parse
import time
from openai import OpenAI

# ================================
# Storage Backend - S3, GCS, or Local
# ================================
STORAGE_BACKEND = os.environ.get('STORAGE_BACKEND', 'local').lower()

s3_client = None
gcs_client = None

if STORAGE_BACKEND == 's3':
    AWS_ACCESS_KEY = st.secrets.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET = st.secrets.get("S3_BUCKET_NAME")
    S3_REGION = st.secrets.get("S3_REGION", "us-east-1")
    if AWS_ACCESS_KEY and AWS_SECRET_KEY and S3_BUCKET:
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=S3_REGION
            )
            s3_client.head_bucket(Bucket=S3_BUCKET)
            st.success("Connected to AWS S3.")
        except Exception as e:
            st.warning(f"S3 connection failed: {e}. Falling back to local.")
            STORAGE_BACKEND = 'local'

elif STORAGE_BACKEND == 'gcs':
    GCS_BUCKET = st.secrets.get("GCS_BUCKET_NAME")
    GCS_CREDENTIALS = st.secrets.get("GCS_CREDENTIALS_JSON")
    if GCS_BUCKET:
        try:
            creds_dict = json.loads(GCS_CREDENTIALS) if GCS_CREDENTIALS else None
            gcs_client = storage.Client.from_service_account_info(creds_dict) if creds_dict else storage.Client()
            bucket = gcs_client.bucket(GCS_BUCKET)
            bucket.blob("test").upload_from_string("test")
            st.success("Connected to Google Cloud Storage.")
        except Exception as e:
            st.warning(f"GCS connection failed: {e}. Falling back to local.")
            STORAGE_BACKEND = 'local'

if STORAGE_BACKEND == 'local':
    LOCAL_UPLOAD_DIR = "uploaded_documents"
    os.makedirs(LOCAL_UPLOAD_DIR, exist_ok=True)

def upload_file_to_storage(file):
    """Upload file to configured storage backend and return public URL or local path."""
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.name}"

    if STORAGE_BACKEND == 's3' and s3_client:
        try:
            s3_client.upload_fileobj(file, S3_BUCKET, filename)
            url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
            return url, True
        except Exception as e:
            st.error(f"S3 upload failed: {e}")
            return None, False
    elif STORAGE_BACKEND == 'gcs' and gcs_client:
        try:
            bucket = gcs_client.bucket(GCS_BUCKET)
            blob = bucket.blob(filename)
            blob.upload_from_file(file, rewind=True)
            blob.make_public()
            url = blob.public_url
            return url, True
        except Exception as e:
            st.error(f"GCS upload failed: {e}")
            return None, False
    else:
        file_path = os.path.join(LOCAL_UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return file_path, False

# ================================
# Database Configuration - Multi-Deal Support
# ================================
DB_TYPE = os.environ.get('DB_TYPE', 'sqlite').lower()
if DB_TYPE == 'sqlite':
    DB_URL = os.environ.get('SQLITE_URL', 'sqlite:///vc_portal.db')
else:
    DB_URL = os.environ.get('POSTGRES_URL')
    if not DB_URL:
        st.error("For PostgreSQL mode, set POSTGRES_URL environment variable.")
        st.stop()

engine = create_engine(DB_URL, echo=False, pool_pre_ping=True, future=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    name = Column(String(100))
    role = Column(String(20), default='analyst')
    deals = relationship("Deal", back_populates="user")

class Deal(Base):
    __tablename__ = 'deals'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    company_name = Column(String(200), nullable=False)
    stage = Column(String(50), default="Sourced")
    created_date = Column(DateTime, default=datetime.datetime.utcnow)
    notes = Column(Text)
    user = relationship("User", back_populates="deals")
    analyses = relationship("Analysis", back_populates="deal")

class Analysis(Base):
    __tablename__ = 'analyses'
    id = Column(Integer, primary_key=True)
    deal_id = Column(Integer, ForeignKey('deals.id'))
    user_id = Column(Integer, ForeignKey('users.id'))
    section = Column(String(100), nullable=False)
    data = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    deal = relationship("Deal", back_populates="analyses")

Base.metadata.create_all(engine)  # Creates tables if they don't exist (no drop_all to preserve data)

# ================================
# Authentication
# ================================
auth_config = {
    "credentials": {
        "usernames": {
            "admin": {
                "email": "admin@example.com",
                "name": "Admin User",
                "password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
            },
            "analyst": {
                "email": "analyst@example.com",
                "name": "Analyst User",
                "password": "$2b$12$KIXp1pa8kT2P/QN1jJ1QBu6s9K2fY0jN3VqO/0oV5P2X4zX1X6X2"
            }
        }
    },
    "cookie": {
        "expiry_days": 30,
        "key": "vc_portal_random_key",
        "name": "vc_portal_cookie"
    }
}

authenticator = Authenticate(
    auth_config["credentials"],
    auth_config["cookie"]["name"],
    auth_config["cookie"]["key"],
    auth_config["cookie"]["expiry_days"]
)

authenticator.login(location="main")

name = st.session_state.get('name')
authentication_status = st.session_state.get('authentication_status')
username = st.session_state.get('username')

if authentication_status:
    st.success(f'Welcome *{name}*')
    authenticator.logout('Logout', 'sidebar')

    session_db = Session()
    user = session_db.query(User).filter_by(username=username).first()
    if not user:
        user = User(username=username, name=name, role='admin' if username == 'admin' else 'analyst')
        session_db.add(user)
        session_db.commit()

    st.set_page_config(page_title="Ultimate VC Due Diligence Portal", layout="wide")
    st.title("🚀 Ultimate Venture Capital Due Diligence Portal – Version 2.0")

    # Multi-Deal State
    if "current_deal_id" not in st.session_state:
        st.session_state.current_deal_id = None

    pages = [
        "Deals Dashboard",
        "Deal Sourcing (AI-Powered)",
        "Financial Due Diligence",
        "Legal Due Diligence",
        "Technical Due Diligence",
        "Operational Due Diligence",
        "Market Due Diligence",
        "Commercial Due Diligence",
        "Team & Founder Analysis",
        "ESG & Responsible AI Check",
        "Term Sheet Negotiation",
        "Financial Model",
        "DCF Valuation",
        "Reverse DCF",
        "Comparable Analysis",
        "Market & Competitor Benchmarking",
        "Sensitivity Analysis",
        "Monte Carlo Simulation",
        "Scenario Planning",
        "Generate Report"
    ]

    sidebar_selection = st.sidebar.selectbox("Navigate", pages)

    # Load current deal
    current_deal = None
    if st.session_state.current_deal_id:
        current_deal = session_db.query(Deal).filter_by(id=st.session_state.current_deal_id, user_id=user.id).first()
        if current_deal:
            st.sidebar.success(f"Active Deal: **{current_deal.company_name}** ({current_deal.stage})")

    # OpenAI Client for AI Summaries
    openai_client = None
    if "OPENAI_API_KEY" in st.secrets and st.secrets.get("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Generic save function
    def save_analysis(section, save_data):
        new_analysis = Analysis(deal_id=current_deal.id, user_id=user.id, section=section, data=json.dumps(save_data))
        session_db.add(new_analysis)
        session_db.commit()

    def calculate_saas_metrics(mrr_start, mrr_end, new_mrr, lost_mrr, expansion_mrr, cac, monthly_burn, gross_margin_percent, num_customers):
        metrics = {}
        metrics["ARR (Current)"] = mrr_end * 12
        net_new_mrr = new_mrr + expansion_mrr - lost_mrr
        metrics["Net New MRR"] = net_new_mrr
        metrics["Gross MRR Churn Rate (%)"] = (lost_mrr / mrr_start * 100) if mrr_start > 0 else 0
        metrics["Net MRR Churn Rate (%)"] = ((lost_mrr - expansion_mrr) / mrr_start * 100) if mrr_start > 0 else 0
        metrics["MRR Growth Rate (%)"] = ((mrr_end - mrr_start) / mrr_start * 100) if mrr_start > 0 else 0
        metrics["Gross Margin (%)"] = gross_margin_percent
        metrics["Rule of 40 Score"] = metrics["MRR Growth Rate (%)"] + gross_margin_percent
        metrics["Magic Number"] = net_new_mrr / cac if cac > 0 else 0
        arpu = mrr_end / num_customers if num_customers > 0 else 0
        gross_margin = gross_margin_percent / 100
        churn_rate = lost_mrr / mrr_start if mrr_start > 0 else 0
        cltv = (arpu * gross_margin) / churn_rate if churn_rate > 0 else 0
        metrics["CLTV ($)"] = cltv
        metrics["LTV/CAC Ratio"] = cltv / cac if cac > 0 else 0
        metrics["Burn Multiple"] = monthly_burn / net_new_mrr if net_new_mrr > 0 else 999
        metrics["Runway (months)"] = (mrr_end * 12) / monthly_burn if monthly_burn > 0 else 999
        metrics["ARPA ($)"] = mrr_end / num_customers if num_customers > 0 else 0
        metrics["Cohort Retention Rate (%)"] = (1 - churn_rate) * 100 if churn_rate < 1 else 0
        metrics["CAC Payback Period (months)"] = cac / (metrics["ARPA ($)"] * gross_margin) if (metrics["ARPA ($)"] * gross_margin) > 0 else 999
        quick_ratio = (new_mrr + expansion_mrr) / lost_mrr if lost_mrr > 0 else 999
        metrics["Quick Ratio"] = quick_ratio
        return metrics

    def ai_analysis(section, inputs):
        score = 0
        risk_level = "High"
        if section == "Financial DD":
            if inputs.get('runway', 0) >= 18: score += 30
            elif inputs.get('runway', 0) >= 12: score += 20
            elif inputs.get('runway', 0) >= 6: score += 10
            if inputs.get('cac_ltv_ratio', 0) >= 3: score += 30
            elif inputs.get('cac_ltv_ratio', 0) >= 2: score += 15
            if inputs.get('gross_margin', 0) >= 70: score += 20
            elif inputs.get('gross_margin', 0) >= 50: score += 10
            if inputs.get('burn_multiple', 999) <= 1.5: score += 20
        elif section in ["Legal DD", "Technical DD", "Operational DD", "Market DD", "Commercial DD"]:
            issues = sum(1 for v in inputs.values() if v)
            score = max(0, 100 - issues * 15)
        elif section == "Team Analysis":
            score = (inputs.get('founder_experience', 0) / 20 * 20) + \
                    (inputs.get('team_completeness', 0) / 100 * 30) + \
                    (inputs.get('references', 0) / 10 * 30) + \
                    (20 if not inputs.get('red_flags', False) else 0)
        elif section == "ESG":
            score = sum(inputs.values()) * 20
        if score >= 80:
            risk_level = "Low"
        elif score >= 50:
            risk_level = "Medium"
        return score, risk_level, f"**AI Risk Score:** {score}/100 | **Risk Level:** {risk_level}"

    # Deals Dashboard
    if sidebar_selection == "Deals Dashboard":
        st.header("📊 Deals Dashboard")
        deals = session_db.query(Deal).filter_by(user_id=user.id).order_by(Deal.created_date.desc()).all()

        if not deals:
            st.info("No deals yet. Create your first deal!")

        st.subheader("Your Deals")
        for deal in deals:
            analyses = session_db.query(Analysis).filter_by(deal_id=deal.id).all()
            completed = len(set(a.section for a in analyses))
            total_sections = 9
            completion = (completed / total_sections) * 100 if total_sections else 0
            scores = []
            for a in analyses:
                if a.data:
                    try:
                        data = json.loads(a.data)
                        if "score" in data:
                            scores.append(data["score"])
                    except:
                        pass
            avg_score = np.mean(scores) if scores else 50
            risk = "Low" if avg_score >= 80 else "Medium" if avg_score >= 50 else "High"

            with st.expander(f"{deal.company_name} — {deal.stage} — Risk: {risk} — {completion:.0f}% Complete"):
                col1, col2, col3 = st.columns(3)
                col1.write(f"Created: {deal.created_date.date()}")
                col2.write(f"Last Updated: {max((a.timestamp for a in analyses), default=deal.created_date).date()}")
                col3.metric("Avg Risk Score", f"{avg_score:.0f}/100")
                if st.button("Select Deal", key=f"select_{deal.id}"):
                    st.session_state.current_deal_id = deal.id
                    st.rerun()
                if st.button("Delete Deal", key=f"delete_{deal.id}"):
                    session_db.query(Analysis).filter_by(deal_id=deal.id).delete()
                    session_db.query(Deal).filter_by(id=deal.id).delete()
                    session_db.commit()
                    if st.session_state.current_deal_id == deal.id:
                        st.session_state.current_deal_id = None
                    st.rerun()

        st.markdown("### Create New Deal")
        with st.form(key="create_deal_form"):
            company_name = st.text_input("Company Name")
            stage = st.selectbox("Initial Stage", ["Sourced", "In DD", "Term Sheet", "Closed", "Passed"])
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Create Deal")
            if submitted:
                if company_name.strip():
                    new_deal = Deal(user_id=user.id, company_name=company_name.strip(), stage=stage, notes=notes)
                    session_db.add(new_deal)
                    session_db.commit()
                    st.session_state.current_deal_id = new_deal.id
                    st.success(f"Deal '{company_name}' created successfully!")
                    st.rerun()
                else:
                    st.error("Company Name is required.")

        st.markdown("### Portfolio Overview (Closed Deals)")
        closed_deals = [d for d in deals if d.stage == "Closed"]
        if closed_deals:
            portfolio_data = []
            for d in closed_deals:
                scores = []
                for a in session_db.query(Analysis).filter_by(deal_id=d.id).all():
                    if a.data:
                        try:
                            data = json.loads(a.data)
                            if "score" in data:
                                scores.append(data["score"])
                        except:
                            pass
                avg = np.mean(scores) if scores else 0
                portfolio_data.append({"Company": d.company_name, "Avg Score": avg, "Created": d.created_date.date()})
            df_portfolio = pd.DataFrame(portfolio_data)
            st.dataframe(df_portfolio)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_portfolio.to_excel(writer, index=False)
            buffer.seek(0)
            st.download_button("Download Portfolio Summary", buffer, "portfolio_summary.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("No closed deals yet.")

    else:
        if not current_deal:
            st.warning("Please select or create a deal from the Deals Dashboard first.")
            st.stop()

        # Deal Sourcing (AI-Powered)
        if sidebar_selection == "Deal Sourcing (AI-Powered)":
            st.header("🔍 Deal Sourcing")
            st.markdown(f"**Current Deal:** {current_deal.company_name}")

            sourcing_method = st.radio("Sourcing Method", (
                "Crunchbase Search",
                "AngelList Search",
                "LinkedIn Company Search",
                "Google Search",
                "OpenAI GPT-4o Search",
                "Google Gemini Search",
                "xAI Grok Search",
                "Manual Entry"
            ))

            uploaded_files = st.file_uploader(
                "Upload Deal Documents (Pitch Deck, Financials, Cap Table, Legal Docs, etc.)",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'xlsx', 'pptx', 'jpg', 'png', 'csv']
            )

            file_urls = []
            if uploaded_files:
                with st.spinner(f"Uploading {len(uploaded_files)} file(s) to {STORAGE_BACKEND.upper()}..."):
                    success_count = 0
                    for file in uploaded_files:
                        if file.size > 50 * 1024 * 1024:
                            st.error(f"File {file.name} is too large (>50MB). Skipping.")
                            continue
                        url, success = upload_file_to_storage(file)
                        if success:
                            file_urls.append({"name": file.name, "url": url})
                            success_count += 1
                        else:
                            file_urls.append({"name": file.name, "url": "Upload failed"})
                    if success_count > 0:
                        st.success(f"{success_count}/{len(uploaded_files)} files uploaded successfully.")
                    else:
                        st.error("All uploads failed.")

            # Key loading
            openai_client = None
            if "OPENAI_API_KEY" in st.secrets and st.secrets.get("OPENAI_API_KEY"):
                openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                st.success("OpenAI API key loaded from Streamlit secrets.")
            elif os.getenv("OPENAI_API_KEY"):
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                st.success("OpenAI API key loaded from environment variable.")
            else:
                st.warning("OpenAI API key not found. Add it to .streamlit/secrets.toml or set as environment variable to enable GPT analysis.")

            gemini_api_key = None
            if "GEMINI_API_KEY" in st.secrets and st.secrets.get("GEMINI_API_KEY"):
                gemini_api_key = st.secrets["GEMINI_API_KEY"]
                st.success("Google Gemini API key loaded from Streamlit secrets.")
            elif os.getenv("GEMINI_API_KEY"):
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                st.success("Google Gemini API key loaded from environment variable.")
            else:
                st.warning("Google Gemini API key not found. Add it to .streamlit/secrets.toml or set as environment variable to enable Gemini search.")

            grok_api_key = None
            if "GROK_API_KEY" in st.secrets and st.secrets.get("GROK_API_KEY"):
                grok_api_key = st.secrets["GROK_API_KEY"]
                st.success("xAI Grok API key loaded from Streamlit secrets.")
            elif os.getenv("GROK_API_KEY"):
                grok_api_key = os.getenv("GROK_API_KEY")
                st.success("xAI Grok API key loaded from environment variable.")
            else:
                st.warning("xAI Grok API key not found. Add it to .streamlit/secrets.toml or set as environment variable to enable Grok search.")

            # OAuth Callback Handling
            query_params = st.query_params
            code = query_params.get("code")

            if code:
                st.info("OAuth callback detected. Processing token...")
                service = st.session_state.get('oauth_service')
                if service == "linkedin":
                    client_id = st.secrets.get("LINKEDIN_CLIENT_ID")
                    client_secret = st.secrets.get("LINKEDIN_CLIENT_SECRET")
                    redirect_uri = st.secrets.get("LINKEDIN_REDIRECT_URI")
                    try:
                        token_url = "https://www.linkedin.com/oauth/v2/accessToken"
                        token_data = {
                            "grant_type": "authorization_code",
                            "code": code,
                            "redirect_uri": redirect_uri,
                            "client_id": client_id,
                            "client_secret": client_secret
                        }
                        response = requests.post(token_url, data=token_data)
                        response.raise_for_status()
                        token_json = response.json()
                        access_token = token_json.get("access_token")
                        expires_in = token_json.get("expires_in", 3600)
                        refresh_token = token_json.get("refresh_token")
                        st.session_state['linkedin_token'] = access_token
                        st.session_state['linkedin_expires'] = time.time() + expires_in
                        if refresh_token:
                            st.session_state['linkedin_refresh'] = refresh_token
                        st.success("LinkedIn token obtained.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"LinkedIn token exchange failed: {e}")
                elif service == "angellist":
                    client_id = st.secrets.get("ANGELLIST_CLIENT_ID")
                    client_secret = st.secrets.get("ANGELLIST_CLIENT_SECRET")
                    redirect_uri = st.secrets.get("ANGELLIST_REDIRECT_URI")
                    try:
                        token_url = "https://angel.co/api/oauth/token"
                        token_data = {
                            "grant_type": "authorization_code",
                            "code": code,
                            "redirect_uri": redirect_uri,
                            "client_id": client_id,
                            "client_secret": client_secret
                        }
                        response = requests.post(token_url, data=token_data)
                        response.raise_for_status()
                        token_json = response.json()
                        access_token = token_json.get("access_token")
                        expires_in = token_json.get("expires_in", 3600)
                        st.session_state['angellist_token'] = access_token
                        st.session_state['angellist_expires'] = time.time() + expires_in
                        st.success("AngelList token obtained.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"AngelList token exchange failed: {e}")
                elif service == "google":
                    client_id = st.secrets.get("GOOGLE_CLIENT_ID")
                    client_secret = st.secrets.get("GOOGLE_CLIENT_SECRET")
                    redirect_uri = st.secrets.get("GOOGLE_REDIRECT_URI")
                    try:
                        token_url = "https://oauth2.googleapis.com/token"
                        token_data = {
                            "grant_type": "authorization_code",
                            "code": code,
                            "redirect_uri": redirect_uri,
                            "client_id": client_id,
                            "client_secret": client_secret
                        }
                        response = requests.post(token_url, data=token_data)
                        response.raise_for_status()
                        token_json = response.json()
                        access_token = token_json.get("access_token")
                        expires_in = token_json.get("expires_in", 3600)
                        refresh_token = token_json.get("refresh_token")
                        st.session_state['google_token'] = access_token
                        st.session_state['google_expires'] = time.time() + expires_in
                        if refresh_token:
                            st.session_state['google_refresh'] = refresh_token
                        st.success("Google token obtained.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Google token exchange failed: {e}")

            # Token Expiry Check Helper
            def is_token_expired(expires_key):
                expires = st.session_state.get(expires_key)
                if expires and time.time() > expires:
                    return True
                return False

            # Token Refresh Helper
            def refresh_token(service):
                if service == "linkedin":
                    client_id = st.secrets.get("LINKEDIN_CLIENT_ID")
                    client_secret = st.secrets.get("LINKEDIN_CLIENT_SECRET")
                    refresh_token = st.session_state.get('linkedin_refresh')
                    if not refresh_token:
                        st.error("No refresh token. Re-authenticate.")
                        return False
                    try:
                        refresh_url = "https://www.linkedin.com/oauth/v2/accessToken"
                        refresh_data = {
                            "grant_type": "refresh_token",
                            "refresh_token": refresh_token,
                            "client_id": client_id,
                            "client_secret": client_secret
                        }
                        response = requests.post(refresh_url, data=refresh_data)
                        response.raise_for_status()
                        token_json = response.json()
                        new_token = token_json.get("access_token")
                        new_expires = time.time() + token_json.get("expires_in", 3600)
                        new_refresh = token_json.get("refresh_token", refresh_token)
                        st.session_state['linkedin_token'] = new_token
                        st.session_state['linkedin_expires'] = new_expires
                        st.session_state['linkedin_refresh'] = new_refresh
                        st.success("LinkedIn token refreshed.")
                        return True
                    except Exception as e:
                        st.error(f"Refresh failed: {e}")
                        return False
                elif service == "angellist":
                    st.info("AngelList does not support refresh tokens. Re-authenticate.")
                    return False
                elif service == "google":
                    client_id = st.secrets.get("GOOGLE_CLIENT_ID")
                    client_secret = st.secrets.get("GOOGLE_CLIENT_SECRET")
                    refresh_token = st.session_state.get('google_refresh')
                    if not refresh_token:
                        st.error("No refresh token. Re-authenticate.")
                        return False
                    try:
                        refresh_url = "https://oauth2.googleapis.com/token"
                        refresh_data = {
                            "grant_type": "refresh_token",
                            "refresh_token": refresh_token,
                            "client_id": client_id,
                            "client_secret": client_secret
                        }
                        response = requests.post(refresh_url, data=refresh_data)
                        response.raise_for_status()
                        token_json = response.json()
                        new_token = token_json.get("access_token")
                        new_expires = time.time() + token_json.get("expires_in", 3600)
                        st.session_state['google_token'] = new_token
                        st.session_state['google_expires'] = new_expires
                        st.success("Google token refreshed.")
                        return True
                    except Exception as e:
                        st.error(f"Google refresh failed: {e}")
                        return False

            if sourcing_method == "Crunchbase Search":
                st.subheader("Crunchbase Search")
                api_key = st.text_input("Crunchbase API Key", type="password", value=st.secrets.get("CRUNCHBASE_API_KEY", ""))
                query = st.text_input("Search query")
                if st.button("Search Crunchbase"):
                    if not api_key:
                        st.error("Crunchbase API key required.")
                    elif not query.strip():
                        st.error("Search query cannot be empty.")
                    else:
                        try:
                            url = "https://api.crunchbase.com/api/v4/searches/organizations"
                            headers = {"X-cb-user-key": api_key}
                            params = {"query": query, "limit": 10}
                            response = requests.get(url, headers=headers, params=params, timeout=30)
                            response.raise_for_status()
                            data = response.json()
                            results = []
                            for e in data.get('entities', []):
                                org = e.get('properties', {})
                                results.append({
                                    "Name": org.get('name'),
                                    "Location": org.get('hq_location'),
                                    "Funding": org.get('total_funding_usd'),
                                    "Description": org.get('short_description')
                                })
                            df = pd.DataFrame(results)
                            st.dataframe(df)
                            deal_data = {
                                "method": "Crunchbase Search",
                                "results": results,
                                "documents": file_urls
                            }
                            save_analysis("Deal Sourcing", deal_data)
                            st.success("Deal saved with documents.")
                        except requests.exceptions.HTTPError as e:
                            if e.response.status_code == 401:
                                st.error("Crunchbase API key invalid or expired. Get a new key.")
                            elif e.response.status_code == 429:
                                st.error("Crunchbase rate limit exceeded. Wait and try again later.")
                            else:
                                st.error(f"HTTP error: {e.response.status_code}")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Network error: {e}")
                        except Exception as e:
                            st.error(f"Unexpected error: {e}")

            elif sourcing_method == "AngelList Search":
                st.subheader("AngelList (Wellfound) Search")
                client_id = st.text_input("AngelList Client ID", type="password", value=st.secrets.get("ANGELLIST_CLIENT_ID", ""))
                client_secret = st.text_input("AngelList Client Secret", type="password", value=st.secrets.get("ANGELLIST_CLIENT_SECRET", ""))
                redirect_uri = st.secrets.get("ANGELLIST_REDIRECT_URI", "http://localhost:8501")
                if st.button("Start AngelList OAuth"):
                    if not client_id or not client_secret:
                        st.error("Client ID and Secret required.")
                    else:
                        st.session_state['oauth_service'] = "angellist"
                        auth_url = f"https://angel.co/api/oauth/authorize?client_id={client_id}&redirect_uri={urllib.parse.quote(redirect_uri)}&response_type=code&scope=read:companies"
                        st.markdown(f"[Click to authorize AngelList]({auth_url})")
                if is_token_expired('angellist_expires'):
                    st.warning("AngelList token expired. Re-authenticate.")
                angellist_token = st.session_state.get('angellist_token', st.text_input("AngelList Access Token", type="password"))
                company_query = st.text_input("Company name or keyword")
                if st.button("Search AngelList"):
                    if angellist_token and company_query:
                        if is_token_expired('angellist_expires'):
                            st.error("Token expired. Re-authenticate.")
                        else:
                            try:
                                url = "https://api.wellfound.com/v1/companies/search"
                                headers = {"Authorization": f"Bearer {angellist_token}"}
                                params = {"q": company_query}
                                response = requests.get(url, headers=headers, params=params)
                                response.raise_for_status()
                                data = response.json()
                                results = []
                                for company in data.get('companies', []):
                                    results.append({
                                        "Name": company.get('name'),
                                        "Description": company.get('product_desc'),
                                        "Funding": company.get('high_concept'),
                                        "Website": company.get('company_url')
                                    })
                                df = pd.DataFrame(results)
                                st.dataframe(df)
                                deal_data = {
                                    "method": "AngelList Search",
                                    "results": results,
                                    "documents": file_urls
                                }
                                save_analysis("Deal Sourcing", deal_data)
                                st.success("AngelList deal saved.")
                            except requests.exceptions.HTTPError as e:
                                if e.response.status_code == 401:
                                    st.error("Token expired or invalid. Re-authenticate.")
                                elif e.response.status_code == 429:
                                    st.error("AngelList rate limit exceeded. Try again later.")
                                else:
                                    st.error(f"HTTP error: {e.response.status_code}")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Network error: {e}")
                            except Exception as e:
                                st.error(f"Unexpected error: {e}")
                    else:
                        st.warning("Token and query required.")

            elif sourcing_method == "LinkedIn Company Search":
                st.subheader("LinkedIn Company Search")
                client_id = st.text_input("LinkedIn Client ID", type="password", value=st.secrets.get("LINKEDIN_CLIENT_ID", ""))
                client_secret = st.text_input("LinkedIn Client Secret", type="password", value=st.secrets.get("LINKEDIN_CLIENT_SECRET", ""))
                redirect_uri = st.secrets.get("LINKEDIN_REDIRECT_URI", "http://localhost:8501")
                if st.button("Start LinkedIn OAuth"):
                    if not client_id or not client_secret:
                        st.error("Client ID and Secret required.")
                    else:
                        st.session_state['oauth_service'] = "linkedin"
                        auth_url = f"https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id={client_id}&redirect_uri={urllib.parse.quote(redirect_uri)}&scope=r_organization_lookup"
                        st.markdown(f"[Click to authorize LinkedIn]({auth_url})")
                if is_token_expired('linkedin_expires'):
                    st.warning("LinkedIn token expired.")
                    if st.button("Refresh LinkedIn Token"):
                        refresh_token("linkedin")
                linkedin_token = st.session_state.get('linkedin_token', st.text_input("LinkedIn Access Token", type="password"))
                company_vanity = st.text_input("Company Vanity Name or ID (e.g., linkedin)")
                if st.button("Search LinkedIn"):
                    if linkedin_token and company_vanity:
                        if is_token_expired('linkedin_expires'):
                            st.error("Token expired. Refresh or re-authenticate.")
                        else:
                            try:
                                url = f"https://api.linkedin.com/rest/organizations?q=vanityName&vanityName={company_vanity}"
                                headers = {
                                    "Authorization": f"Bearer {linkedin_token}",
                                    "X-Restli-Protocol-Version": "2.0.0",
                                    "Linkedin-Version": "202511"
                                }
                                response = requests.get(url, headers=headers, timeout=30)
                                response.raise_for_status()
                                data = response.json()
                                results = data.get('elements', [])
                                if results:
                                    org = results[0]
                                    result_data = {
                                        "Name": org.get('localizedName'),
                                        "Description": org.get('localizedDescription'),
                                        "Website": org.get('localizedWebsite'),
                                        "Employee Count": org.get('staffCount')
                                    }
                                    st.json(result_data)
                                    deal_data = {
                                        "method": "LinkedIn Search",
                                        "results": [result_data],
                                        "documents": file_urls
                                    }
                                    save_analysis("Deal Sourcing", deal_data)
                                    st.success("LinkedIn company data saved.")
                                else:
                                    st.info("No company found.")
                            except requests.exceptions.HTTPError as e:
                                if e.response.status_code == 401:
                                    st.error("Token expired or invalid. Refresh or re-authenticate.")
                                elif e.response.status_code == 429:
                                    st.error("LinkedIn rate limit exceeded. Try again later.")
                                else:
                                    st.error(f"HTTP error: {e.response.status_code}")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Network error: {e}")
                            except Exception as e:
                                st.error(f"Unexpected error: {e}")
                    else:
                        st.warning("Token and vanity name required.")

            elif sourcing_method == "Google Search":
                st.subheader("Google Custom Search")
                google_api_key = st.text_input("Google API Key", type="password", value=st.secrets.get("GOOGLE_API_KEY", ""))
                google_cx = st.text_input("Google Custom Search Engine ID (CX)", value=st.secrets.get("GOOGLE_CX", ""))
                google_query = st.text_input("Search query")
                if st.button("Search Google"):
                    if not google_api_key or not google_cx:
                        st.error("Google API key and CX required.")
                    elif not google_query.strip():
                        st.error("Search query required.")
                    else:
                        try:
                            url = "https://www.googleapis.com/customsearch/v1"
                            params = {
                                "key": google_api_key,
                                "cx": google_cx,
                                "q": google_query
                            }
                            response = requests.get(url, params=params)
                            response.raise_for_status()
                            data = response.json()
                            results = []
                            for item in data.get('items', []):
                                results.append({
                                    "Title": item.get('title'),
                                    "Link": item.get('link'),
                                    "Snippet": item.get('snippet')
                                })
                            df = pd.DataFrame(results)
                            st.dataframe(df)
                            deal_data = {
                                "method": "Google Search",
                                "results": results,
                                "documents": file_urls
                            }
                            save_analysis("Deal Sourcing", deal_data)
                            st.success("Google search results saved.")
                        except requests.exceptions.HTTPError as e:
                            if e.response.status_code == 401:
                                st.error("Google API key invalid or expired. Get a new key.")
                            elif e.response.status_code == 429:
                                st.error("Google rate limit exceeded. Try again later.")
                            else:
                                st.error(f"HTTP error: {e.response.status_code}")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Network error: {e}")
                        except Exception as e:
                            st.error(f"Unexpected error: {e}")

            elif sourcing_method == "OpenAI GPT-4o Search":
                st.subheader("OpenAI GPT-4o Search")
                if not openai_client:
                    st.warning("OpenAI API key not configured — GPT-4o search disabled.")
                else:
                    ai_query = st.text_area("Describe the type of startup you're looking for", value="AI-powered SaaS companies in healthcare with >$1M ARR founded in the last 2 years")
                    if st.button("Generate Startup Leads with GPT-4o"):
                        try:
                            response = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "You are a VC deal sourcing expert. Generate a list of 10 promising startups matching the user's criteria. Include name, short description, estimated ARR, funding stage, and why it's a good opportunity."},
                                    {"role": "user", "content": ai_query}
                                ],
                                temperature=0.7,
                                max_tokens=1500
                            )
                            gpt_results = response.choices[0].message.content
                            st.markdown("### GPT-4o Generated Leads")
                            st.markdown(gpt_results)
                            deal_data = {
                                "method": "OpenAI GPT-4o Search",
                                "query": ai_query,
                                "results": gpt_results,
                                "documents": file_urls
                            }
                            save_analysis("Deal Sourcing", deal_data)
                            st.success("GPT-4o leads saved.")
                        except Exception as e:
                            st.error(f"OpenAI call failed: {e}")

            elif sourcing_method == "Google Gemini Search":
                st.subheader("Google Gemini Search")
                if not gemini_api_key:
                    st.warning("Google Gemini API key not configured — Gemini search disabled.")
                else:
                    gemini_query = st.text_area("Describe the type of startup or market you're researching", value="Fast-growing fintech startups in Southeast Asia with recent funding")
                    if st.button("Generate Leads with Gemini"):
                        try:
                            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
                            payload = {
                                "contents": [{
                                    "parts": [{
                                        "text": f"You are a VC deal sourcing expert. Generate a list of 10 promising startups matching this criteria: {gemini_query}. Include name, description, estimated ARR, funding stage, and investment rationale."
                                    }]
                                }]
                            }
                            response = requests.post(url, json=payload)
                            response.raise_for_status()
                            data = response.json()
                            gemini_results = data['candidates'][0]['content']['parts'][0]['text']
                            st.markdown("### Gemini Generated Leads")
                            st.markdown(gemini_results)
                            deal_data = {
                                "method": "Google Gemini Search",
                                "query": gemini_query,
                                "results": gemini_results,
                                "documents": file_urls
                            }
                            save_analysis("Deal Sourcing", deal_data)
                            st.success("Gemini leads saved.")
                        except Exception as e:
                            st.error(f"Gemini call failed: {e}")

            elif sourcing_method == "xAI Grok Search":
                st.subheader("xAI Grok Search")
                if not grok_api_key:
                    st.warning("xAI Grok API key not configured — Grok search disabled.")
                else:
                    grok_query = st.text_area("Describe the type of startup or market you're researching", value="Emerging web3 startups in gaming with recent funding")
                    if st.button("Generate Leads with Grok"):
                        try:
                            url = "https://api.x.ai/v1/chat/completions"
                            headers = {
                                "Authorization": f"Bearer {grok_api_key}",
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": "grok-beta",
                                "messages": [
                                    {"role": "system", "content": "You are a VC deal sourcing expert."},
                                    {"role": "user", "content": grok_query}
                                ],
                                "temperature": 0.7,
                                "max_tokens": 1500
                            }
                            response = requests.post(url, headers=headers, json=payload)
                            response.raise_for_status()
                            data = response.json()
                            grok_results = data['choices'][0]['message']['content']
                            st.markdown("### Grok Generated Leads")
                            st.markdown(grok_results)
                            deal_data = {
                                "method": "xAI Grok Search",
                                "query": grok_query,
                                "results": grok_results,
                                "documents": file_urls
                            }
                            save_analysis("Deal Sourcing", deal_data)
                            st.success("Grok leads saved.")
                        except Exception as e:
                            st.error(f"Grok call failed: {e}")

            else:
                st.subheader("Manual Deal Entry")
                with st.form("manual_deal_form"):
                    company_name = st.text_input("Company Name *")
                    description = st.text_area("Description")
                    location = st.text_input("Location")
                    funding = st.number_input("Total Funding Raised ($)", min_value=0.0)
                    stage = st.selectbox("Stage", ["Pre-Seed", "Seed", "Series A", "Series B", "Series C+"])
                    founders = st.text_input("Founders")
                    website = st.text_input("Website")
                    notes = st.text_area("Additional Notes")
                    submitted = st.form_submit_button("Save Manual Deal")
                if submitted:
                    errors = []
                    if not company_name.strip():
                        errors.append("Company Name is required.")
                    if website and not validators.url(website):
                        errors.append("Website must be a valid URL (e.g., https://example.com).")
                    if funding < 0:
                        errors.append("Funding cannot be negative.")
                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        manual_data = {
                            "method": "Manual Entry",
                            "company_name": company_name,
                            "description": description,
                            "location": location,
                            "funding": funding,
                            "stage": stage,
                            "founders": founders,
                            "website": website,
                            "notes": notes,
                            "documents": file_urls
                        }
                        save_analysis("Deal Sourcing", manual_data)
                        st.success("Manual deal saved with uploaded documents.")

        # Financial Due Diligence
        elif sidebar_selection == "Financial Due Diligence":
            st.header("🔍 Financial Due Diligence")
            st.markdown(f"**Deal:** {current_deal.company_name}")

            with st.form("financial_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    mrr_start = st.number_input("Starting MRR ($)", min_value=0.0, value=100000.0)
                    mrr_end = st.number_input("Ending MRR ($)", min_value=0.0, value=120000.0)
                    new_mrr = st.number_input("New MRR from New Customers ($)", min_value=0.0, value=30000.0)
                    lost_mrr = st.number_input("Lost MRR (Churn) ($)", min_value=0.0, value=10000.0)
                with col2:
                    expansion_mrr = st.number_input("Expansion MRR ($)", min_value=0.0, value=5000.0)
                    cac = st.number_input("Customer Acquisition Cost (CAC) ($)", min_value=0.0, value=30000.0)
                    monthly_burn = st.number_input("Monthly Burn Rate ($)", min_value=0.0, value=50000.0)
                with col3:
                    gross_margin_percent = st.number_input("Gross Margin (%)", min_value=0.0, max_value=100.0, value=75.0)
                    num_customers = st.number_input("Number of Customers", min_value=1, value=100)
                submitted = st.form_submit_button("Calculate & Save")

            if submitted:
                saas_metrics = calculate_saas_metrics(
                    mrr_start=mrr_start,
                    mrr_end=mrr_end,
                    new_mrr=new_mrr,
                    lost_mrr=lost_mrr,
                    expansion_mrr=expansion_mrr,
                    cac=cac,
                    monthly_burn=monthly_burn,
                    gross_margin_percent=gross_margin_percent,
                    num_customers=num_customers
                )

                st.subheader("Key SaaS Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current ARR", f"${saas_metrics.get('ARR (Current)', 0):,.0f}")
                col2.metric("Net New MRR", f"${saas_metrics.get('Net New MRR', 0):,.0f}")
                col3.metric("Gross MRR Churn Rate", f"{saas_metrics.get('Gross MRR Churn Rate (%)', 0):.1f}%")
                col4.metric("Net MRR Churn Rate", f"{saas_metrics.get('Net MRR Churn Rate (%)', 0):.1f}%")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MRR Growth Rate", f"{saas_metrics.get('MRR Growth Rate (%)', 0):.1f}%")
                col2.metric("Rule of 40 Score", f"{saas_metrics.get('Rule of 40 Score', 0):.1f}")
                col3.metric("Magic Number", f"{saas_metrics.get('Magic Number', 0):.2f}")
                col4.metric("CLTV ($)", f"${saas_metrics.get('CLTV ($)', 0):,.0f}")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("LTV/CAC Ratio", f"{saas_metrics.get('LTV/CAC Ratio', 0):.1f}")
                col2.metric("Burn Multiple", f"{saas_metrics.get('Burn Multiple', 0):.2f}")
                col3.metric("Runway (months)", f"{saas_metrics.get('Runway (months)', 0):.1f}")
                col4.metric("ARPA ($)", f"${saas_metrics.get('ARPA ($)', 0):,.0f}")

                col1, col2 = st.columns(2)
                col1.metric("Cohort Retention Rate (%)", f"{saas_metrics.get('Cohort Retention Rate (%)', 0):.1f}")
                col2.metric("CAC Payback Period (months)", f"{saas_metrics.get('CAC Payback Period (months)', 0):.1f}")

                col1, col2 = st.columns(2)
                col1.metric("Quick Ratio", f"{saas_metrics.get('Quick Ratio', 0):.2f}")

                inputs = {
                    "runway": saas_metrics.get('Runway (months)', 0),
                    "cac_ltv_ratio": saas_metrics.get('LTV/CAC Ratio', 0),
                    "gross_margin": gross_margin_percent,
                    "burn_multiple": saas_metrics.get('Burn Multiple', 0)
                }
                score, risk, analysis = ai_analysis("Financial DD", inputs)
                st.markdown(analysis)
                st.metric("Financial Health Score", f"{score}/100")

                save_data = {
                    "inputs": {
                        "mrr_start": mrr_start,
                        "mrr_end": mrr_end,
                        "new_mrr": new_mrr,
                        "lost_mrr": lost_mrr,
                        "expansion_mrr": expansion_mrr,
                        "cac": cac,
                        "monthly_burn": monthly_burn,
                        "gross_margin_percent": gross_margin_percent,
                        "num_customers": num_customers
                    },
                    "metrics": saas_metrics,
                    "score": score
                }
                save_analysis("Financial DD", save_data)
                st.success("Financial analysis and SaaS metrics saved.")

                if openai_client and st.button("Generate AI Summary"):
                    summary_prompt = f"Summarize financial health for {current_deal.company_name}: Current ARR ${saas_metrics.get('ARR (Current)', 0):,.0f}, Growth {saas_metrics.get('MRR Growth Rate (%)', 0):.1f}%, Runway {saas_metrics.get('Runway (months)', 0):.1f} months, Score {score}/100"
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "You are a VC analyst."}, {"role": "user", "content": summary_prompt}]
                        )
                        st.markdown("### AI Financial Summary")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI summary failed: {e}")

                df_metrics = pd.DataFrame.from_dict(saas_metrics, orient='index', columns=['Value'])
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_metrics.to_excel(writer, sheet_name='SaaS Metrics')
                buffer.seek(0)
                st.download_button("Export Financial Metrics", buffer, "financial_metrics.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Legal Due Diligence
        elif sidebar_selection == "Legal Due Diligence":
            st.header("⚖️ Legal Due Diligence")
            st.markdown(f"**Deal:** {current_deal.company_name}")

            with st.form("legal_form"):
                st.write("Flag Legal Issues (check if applicable):")
                col1, col2 = st.columns(2)
                with col1:
                    litigation = st.checkbox("Pending or Threatened Litigation")
                    ip_issues = st.checkbox("IP Infringement or Disputes")
                    compliance = st.checkbox("Regulatory Compliance Issues")
                with col2:
                    contracts = st.checkbox("Unfavorable Material Contracts")
                    employees = st.checkbox("Missing Invention Assignments/Non-competes")
                    securities = st.checkbox("Securities/Financing Compliance Issues")
                notes = st.text_area("Additional Legal Notes")
                submitted = st.form_submit_button("Run Legal Risk Analysis")

            if submitted:
                inputs = {
                    "litigation": litigation,
                    "ip_issues": ip_issues,
                    "compliance": compliance,
                    "contracts": contracts,
                    "employees": employees,
                    "securities": securities
                }
                score, risk, analysis = ai_analysis("Legal DD", inputs)
                st.markdown(analysis)
                st.metric("Legal Risk Score", f"{score}/100")
                if score < 70:
                    st.warning("High legal risks detected — recommend thorough review by counsel.")
                save_data = inputs | {"notes": notes, "score": score}
                save_analysis("Legal DD", save_data)
                st.success("Legal analysis saved.")

                if openai_client and st.button("Generate AI Summary"):
                    summary_prompt = f"Summarize legal risks for {current_deal.company_name}: {inputs} Score {score}/100"
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "You are a VC legal expert."}, {"role": "user", "content": summary_prompt}]
                        )
                        st.markdown("### AI Legal Summary")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI summary failed: {e}")

        # Technical Due Diligence
        elif sidebar_selection == "Technical Due Diligence":
            st.header("🛠️ Technical Due Diligence")
            st.markdown(f"**Deal:** {current_deal.company_name}")

            with st.form("technical_form"):
                st.write("Flag Technical Issues (check if applicable):")
                col1, col2 = st.columns(2)
                with col1:
                    tech_stack_obsolete = st.checkbox("Obsolete or Unsupported Tech Stack")
                    scalability_issues = st.checkbox("Scalability Limitations")
                    security_vulnerabilities = st.checkbox("Security Vulnerabilities or Breaches")
                with col2:
                    code_quality_poor = st.checkbox("Poor Code Quality or Technical Debt")
                    ip_ownership_issues = st.checkbox("IP/Tech Ownership Disputes")
                    integration_risks = st.checkbox("High Integration Risks with Third-Parties")
                notes = st.text_area("Additional Technical Notes")
                submitted = st.form_submit_button("Run Technical Risk Analysis")

            if submitted:
                inputs = {
                    "tech_stack_obsolete": tech_stack_obsolete,
                    "scalability_issues": scalability_issues,
                    "security_vulnerabilities": security_vulnerabilities,
                    "code_quality_poor": code_quality_poor,
                    "ip_ownership_issues": ip_ownership_issues,
                    "integration_risks": integration_risks
                }
                score, risk, analysis = ai_analysis("Technical DD", inputs)
                st.markdown(analysis)
                st.metric("Technical Risk Score", f"{score}/100")
                if score < 70:
                    st.warning("High technical risks detected — recommend code audit or tech review.")
                save_data = inputs | {"notes": notes, "score": score}
                save_analysis("Technical DD", save_data)
                st.success("Technical analysis saved.")

                if openai_client and st.button("Generate AI Summary"):
                    summary_prompt = f"Summarize technical risks for {current_deal.company_name}: {inputs} Score {score}/100"
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "You are a VC technical due diligence expert."}, {"role": "user", "content": summary_prompt}]
                        )
                        st.markdown("### AI Technical Summary")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI summary failed: {e}")

        # Operational Due Diligence
        elif sidebar_selection == "Operational Due Diligence":
            st.header("🏢 Operational Due Diligence")
            st.markdown(f"**Deal:** {current_deal.company_name}")

            with st.form("operational_form"):
                st.write("Flag Operational Issues (check if applicable):")
                col1, col2 = st.columns(2)
                with col1:
                    supply_chain_risks = st.checkbox("Supply Chain or Vendor Dependencies Risks")
                    customer_support_weak = st.checkbox("Weak Customer Support Processes")
                    hr_issues = st.checkbox("HR or Talent Retention Issues")
                with col2:
                    risk_management_poor = st.checkbox("Poor Risk Management Framework")
                    compliance_ops = st.checkbox("Operational Compliance Gaps")
                    efficiency_low = st.checkbox("Low Operational Efficiency Metrics")
                notes = st.text_area("Additional Operational Notes")
                submitted = st.form_submit_button("Run Operational Risk Analysis")

            if submitted:
                inputs = {
                    "supply_chain_risks": supply_chain_risks,
                    "customer_support_weak": customer_support_weak,
                    "hr_issues": hr_issues,
                    "risk_management_poor": risk_management_poor,
                    "compliance_ops": compliance_ops,
                    "efficiency_low": efficiency_low
                }
                score, risk, analysis = ai_analysis("Operational DD", inputs)
                st.markdown(analysis)
                st.metric("Operational Risk Score", f"{score}/100")
                if score < 70:
                    st.warning("High operational risks detected — recommend ops audit.")
                save_data = inputs | {"notes": notes, "score": score}
                save_analysis("Operational DD", save_data)
                st.success("Operational analysis saved.")

                if openai_client and st.button("Generate AI Summary"):
                    summary_prompt = f"Summarize operational risks for {current_deal.company_name}: {inputs} Score {score}/100"
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "You are a VC operational expert."}, {"role": "user", "content": summary_prompt}]
                        )
                        st.markdown("### AI Operational Summary")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI summary failed: {e}")

        # Market Due Diligence
        elif sidebar_selection == "Market Due Diligence":
            st.header("🌍 Market Due Diligence")
            st.markdown(f"**Deal:** {current_deal.company_name}")

            with st.form("market_form"):
                st.write("Flag Market Issues (check if applicable):")
                col1, col2 = st.columns(2)
                with col1:
                    market_small = st.checkbox("Small or Niche Market Size")
                    high_competition = st.checkbox("High Competitive Intensity")
                    low_growth = st.checkbox("Low Market Growth Rate")
                with col2:
                    regulatory_risk = st.checkbox("Significant Regulatory Barriers")
                    substitution_risk = st.checkbox("High Risk of Substitution")
                    customer_concentration = st.checkbox("High Customer Concentration Risk")
                notes = st.text_area("Additional Market Notes")
                submitted = st.form_submit_button("Run Market Risk Analysis")

            if submitted:
                inputs = {
                    "market_small": market_small,
                    "high_competition": high_competition,
                    "low_growth": low_growth,
                    "regulatory_risk": regulatory_risk,
                    "substitution_risk": substitution_risk,
                    "customer_concentration": customer_concentration
                }
                score, risk, analysis = ai_analysis("Market DD", inputs)
                st.markdown(analysis)
                st.metric("Market Risk Score", f"{score}/100")
                if score < 70:
                    st.warning("High market risks detected — recommend deeper TAM/SAM analysis.")
                save_data = inputs | {"notes": notes, "score": score}
                save_analysis("Market DD", save_data)
                st.success("Market analysis saved.")

                if openai_client and st.button("Generate AI Summary"):
                    summary_prompt = f"Summarize market risks for {current_deal.company_name}: {inputs} Score {score}/100"
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "You are a VC market expert."}, {"role": "user", "content": summary_prompt}]
                        )
                        st.markdown("### AI Market Summary")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI summary failed: {e}")

        # Commercial Due Diligence
        elif sidebar_selection == "Commercial Due Diligence":
            st.header("💼 Commercial Due Diligence")
            st.markdown(f"**Deal:** {current_deal.company_name}")

            with st.form("commercial_form"):
                st.write("Flag Commercial Issues (check if applicable):")
                col1, col2 = st.columns(2)
                with col1:
                    pricing_weak = st.checkbox("Unclear or Weak Pricing Strategy")
                    sales_cycle_long = st.checkbox("Long or Unpredictable Sales Cycle")
                    channel_issues = st.checkbox("Distribution/Channel Partner Issues")
                with col2:
                    customer_acq_difficult = st.checkbox("High Customer Acquisition Difficulty")
                    low_retention = st.checkbox("Low Customer Retention Indicators")
                    product_market_fit = st.checkbox("Questionable Product-Market Fit")
                notes = st.text_area("Additional Commercial Notes")
                submitted = st.form_submit_button("Run Commercial Risk Analysis")

            if submitted:
                inputs = {
                    "pricing_weak": pricing_weak,
                    "sales_cycle_long": sales_cycle_long,
                    "channel_issues": channel_issues,
                    "customer_acq_difficult": customer_acq_difficult,
                    "low_retention": low_retention,
                    "product_market_fit": product_market_fit
                }
                score, risk, analysis = ai_analysis("Commercial DD", inputs)
                st.markdown(analysis)
                st.metric("Commercial Risk Score", f"{score}/100")
                if score < 70:
                    st.warning("High commercial risks detected — recommend customer interviews and sales pipeline review.")
                save_data = inputs | {"notes": notes, "score": score}
                save_analysis("Commercial DD", save_data)
                st.success("Commercial analysis saved.")

                if openai_client and st.button("Generate AI Summary"):
                    summary_prompt = f"Summarize commercial risks for {current_deal.company_name}: {inputs} Score {score}/100"
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "You are a VC commercial expert."}, {"role": "user", "content": summary_prompt}]
                        )
                        st.markdown("### AI Commercial Summary")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI summary failed: {e}")

        # Team & Founder Analysis
        elif sidebar_selection == "Team & Founder Analysis":
            st.header("👥 Team & Founder Analysis")
            st.markdown(f"**Deal:** {current_deal.company_name}")

            with st.form("team_form"):
                col1, col2 = st.columns(2)
                with col1:
                    founder_experience = st.slider("Founder Experience (years in domain)", 0, 20, 8)
                    references = st.slider("Reference Strength (1-10)", 1, 10, 8)
                with col2:
                    team_completeness = st.slider("Team Completeness (%)", 0, 100, 75)
                    red_flags = st.checkbox("Any Background Red Flags")
                notes = st.text_area("Team Notes")
                submitted = st.form_submit_button("Analyze Team")

            if submitted:
                inputs = {
                    "founder_experience": founder_experience,
                    "team_completeness": team_completeness,
                    "references": references,
                    "red_flags": red_flags
                }
                score, risk, analysis = ai_analysis("Team Analysis", inputs)
                st.markdown(analysis)
                st.metric("Team Score", f"{score}/100")
                if red_flags:
                    st.warning("Background red flags noted — verify thoroughly.")
                save_data = inputs | {"notes": notes, "score": score}
                save_analysis("Team Analysis", save_data)
                st.success("Team analysis saved.")

                if openai_client and st.button("Generate AI Summary"):
                    summary_prompt = f"Summarize team strength for {current_deal.company_name}: {inputs} Score {score}/100"
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "You are a VC team expert."}, {"role": "user", "content": summary_prompt}]
                        )
                        st.markdown("### AI Team Summary")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI summary failed: {e}")

        # ESG & Responsible AI Check
        elif sidebar_selection == "ESG & Responsible AI Check":
            st.header("🌿 ESG & Responsible AI Check")
            st.markdown(f"**Deal:** {current_deal.company_name}")

            with st.form("esg_form"):
                diversity = st.slider("Diversity & Inclusion (1-5)", 1, 5, 3)
                environmental = st.slider("Environmental Impact (1-5)", 1, 5, 3)
                governance = st.slider("Corporate Governance (1-5)", 1, 5, 3)
                ai_ethics = st.slider("AI Ethics & Bias Mitigation (1-5)", 1, 5, 3)
                data_privacy = st.slider("Data Privacy & Security (1-5)", 1, 5, 3)
                notes = st.text_area("ESG Notes")
                submitted = st.form_submit_button("Analyze ESG")

            if submitted:
                inputs = {
                    "diversity": diversity,
                    "environmental": environmental,
                    "governance": governance,
                    "ai_ethics": ai_ethics,
                    "data_privacy": data_privacy
                }
                score, risk, analysis = ai_analysis("ESG", inputs)
                st.markdown(analysis)
                st.metric("ESG Score", f"{score}/100")
                save_data = inputs | {"notes": notes, "score": score}
                save_analysis("ESG", save_data)
                st.success("ESG analysis saved.")

                if openai_client and st.button("Generate AI Summary"):
                    summary_prompt = f"Summarize ESG for {current_deal.company_name}: {inputs} Score {score}/100"
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "You are a VC ESG expert."}, {"role": "user", "content": summary_prompt}]
                        )
                        st.markdown("### AI ESG Summary")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI summary failed: {e}")

        # Term Sheet Negotiation
        elif sidebar_selection == "Term Sheet Negotiation":
            st.header("📄 Term Sheet Negotiation Guide")
            st.markdown(f"**Deal:** {current_deal.company_name}")
            st.markdown("Key terms to negotiate:")
            st.markdown("- Valuation (pre/post-money)")
            st.markdown("- Liquidation Preference (1x non-participating ideal)")
            st.markdown("- Anti-dilution (broad-based weighted average)")
            st.markdown("- Board composition")
            st.markdown("- Protective provisions")
            st.markdown("- Pro-rata rights")

        # Financial Model
        elif sidebar_selection == "Financial Model":
            st.header("📊 Interactive Financial Model")
            st.markdown(f"**Deal:** {current_deal.company_name}")
            with st.expander("Input Assumptions"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    starting_mrr = st.number_input("Starting MRR ($)", value=10000)
                    growth_rate = st.slider("Monthly Growth Rate (%)", 5.0, 30.0, 15.0) / 100
                with col2:
                    churn_rate = st.slider("Monthly Churn (%)", 1.0, 10.0, 4.0) / 100
                    gross_margin = st.slider("Gross Margin (%)", 50.0, 90.0, 75.0) / 100
                with col3:
                    opex_monthly = st.number_input("Monthly OpEx ($)", value=50000)
            if st.button("Generate 36-Month Model"):
                months = 36
                mrr = [starting_mrr]
                for i in range(1, months):
                    mrr.append(mrr[-1] * (1 + growth_rate) * (1 - churn_rate))
                df = pd.DataFrame({
                    'Month': range(1, months+1),
                    'MRR': mrr,
                    'ARR': [x*12 for x in mrr],
                    'Gross Profit': [x * gross_margin for x in mrr],
                    'OpEx': [opex_monthly] * months
                })
                df['Net Income'] = df['Gross Profit'] - df['OpEx']
                st.dataframe(df.style.format({'MRR': '${:,.0f}', 'ARR': '${:,.0f}'}))
                chart = alt.Chart(df).mark_line().encode(x='Month', y='ARR')
                st.altair_chart(chart, use_container_width=True)
                save_data = df.to_dict()
                save_analysis("Financial Model", save_data)
                st.success("Financial Model saved.")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                output.seek(0)
                st.download_button("Download Model", output, "financial_model.xlsx")

        # DCF Valuation
        elif sidebar_selection == "DCF Valuation":
            st.header("💰 DCF Valuation")
            st.markdown(f"**Deal:** {current_deal.company_name}")
            fcff_input = st.text_input("FCFF Years 1-5 ($k)", value="-450,-250,100,900,1925")
            discount_rate = st.number_input("Discount Rate (%)", value=35.0) / 100
            terminal_growth = st.number_input("Terminal Growth (%)", value=3.0) / 100
            if st.button("Calculate DCF"):
                fcff_list = [float(x) for x in fcff_input.split(',')]
                pv_explicit = sum(fcff_list[i] / (1 + discount_rate)**(i+1) for i in range(5))
                year6 = fcff_list[4] * (1 + terminal_growth)
                tv = year6 / (discount_rate - terminal_growth) if discount_rate > terminal_growth else 0
                pv_tv = tv / (1 + discount_rate)**5
                ev = pv_explicit + pv_tv
                st.success(f"Enterprise Value: ${ev:,.0f}k")
                df = pd.DataFrame({'Year': [1,2,3,4,5,'Terminal'], 'FCFF': fcff_list + [year6], 'PV': [fcff_list[i] / (1 + discount_rate)**(i+1) for i in range(5)] + [pv_tv]})
                st.dataframe(df)
                save_data = {"ev": ev, "data": df.to_dict()}
                save_analysis("DCF", save_data)
                st.success("DCF Valuation saved.")
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                buffer.seek(0)
                st.download_button("Export DCF", buffer, "dcf_valuation.xlsx")

        # Reverse DCF
        elif sidebar_selection == "Reverse DCF":
            st.header("🔄 Reverse DCF")
            st.markdown(f"**Deal:** {current_deal.company_name}")
            target_ev = st.number_input("Target Enterprise Value ($k)", value=2500)
            discount_rate = st.number_input("Discount Rate (%)", value=35.0) / 100
            fcff_input = st.text_input("FCFF Years 1-5 ($k)", value="-450,-250,100,900,1925")
            if st.button("Calculate Implied Growth"):
                fcff_list = [float(x) for x in fcff_input.split(',')]
                pv_explicit = sum(fcff_list[i] / (1 + discount_rate)**(i+1) for i in range(5))
                remaining = target_ev - pv_explicit
                tv = remaining * (1 + discount_rate)**5
                implied_g = fsolve(lambda g: fcff_list[4] * (1 + g) / (discount_rate - g) - tv, 0.04)[0]
                st.write(f"Implied Terminal Growth: {implied_g * 100:.2f}%")
                save_data = {"implied_growth": implied_g}
                save_analysis("Reverse DCF", save_data)
                st.success("Reverse DCF saved.")

        # Comparable Analysis
        elif sidebar_selection == "Comparable Analysis":
            st.header("📈 Comparable Company Analysis")
            st.markdown(f"**Deal:** {current_deal.company_name}")
            with st.form("comps_form"):
                col1, col2 = st.columns(2)
                with col1:
                    your_arr = st.number_input("Your ARR ($M)", min_value=0.0, value=6.0)
                with col2:
                    peer_multiples = st.text_input("Peer EV/ARR Multiples (comma-separated)", value="8.0,6.5,9.2,7.8")
                submitted = st.form_submit_button("Run Comps Analysis")

            if submitted:
                try:
                    peer_multiples_list = [float(x) for x in peer_multiples.split(',')]
                except ValueError:
                    st.error("Invalid input for multiples - use numbers separated by commas.")
                else:
                    median_multiple = np.median(peer_multiples_list)
                    your_ev = your_arr * median_multiple
                    st.metric("Median Peer Multiple", f"{median_multiple:.1f}x")
                    st.metric("Your Implied EV ($M)", f"${your_ev:.1f}")
                    save_data = {
                        "your_arr": your_arr,
                        "median_multiple": median_multiple,
                        "your_ev": your_ev
                    }
                    save_analysis("Comparable Analysis", save_data)
                    st.success("Comps analysis saved.")

        # Market & Competitor Benchmarking
        elif sidebar_selection == "Market & Competitor Benchmarking":
            st.header("📊 Market & Competitor Benchmarking")
            st.markdown(f"**Deal:** {current_deal.company_name}")
            your_arr = st.number_input("Your ARR ($M)", value=6.0)
            peer_arrs = st.text_input("Peer ARRs ($M, comma-separated)", value="10,15,8,20")
            if st.button("Analyze"):
                try:
                    peers = [float(x) for x in peer_arrs.split(',')]
                    rank = sum(your_arr > p for p in peers) + 1
                    share = your_arr / (your_arr + sum(peers)) * 100
                    st.metric("Estimated Market Share", f"{share:.1f}%")
                    st.metric("Peer Rank", f"#{rank}")
                    save_data = {"share": share, "rank": rank}
                    save_analysis("Market Benchmark", save_data)
                    st.success("Benchmarking saved.")
                except:
                    st.error("Invalid peer ARR input.")

        # Sensitivity Analysis
        elif sidebar_selection == "Sensitivity Analysis":
            st.header("📉 Sensitivity Analysis")
            st.markdown(f"**Deal:** {current_deal.company_name}")
            fcff_list = [-450, -250, 100, 900, 1925]
            results = []
            for r in np.linspace(0.25, 0.50, 11):
                for g in np.linspace(0.01, 0.06, 6):
                    if r <= g: continue
                    pv_explicit = sum(fcff_list[i] / (1 + r)**(i+1) for i in range(5))
                    tv = fcff_list[4] * (1 + g) / (r - g)
                    pv_tv = tv / (1 + r)**5
                    ev = pv_explicit + pv_tv
                    results.append([f"{r*100:.0f}%", f"{g*100:.0f}%", round(ev)])
            df = pd.DataFrame(results, columns=["Discount Rate", "Terminal Growth", "EV ($k)"])
            pivot = df.pivot(index="Discount Rate", columns="Terminal Growth", values="EV ($k)")
            st.dataframe(pivot.style.background_gradient())
            save_data = pivot.to_dict()
            save_analysis("Sensitivity", save_data)
            st.success("Sensitivity analysis saved.")

        # Monte Carlo Simulation
        elif sidebar_selection == "Monte Carlo Simulation":
            st.header("🎲 Monte Carlo Simulation")
            st.markdown(f"**Deal:** {current_deal.company_name}")
            simulations = st.number_input("Number of Simulations", 1000, 10000, 5000)
            mean_discount = st.slider("Mean Discount Rate (%)", 25.0, 50.0, 35.0) / 100
            std_discount = st.slider("Std Dev Discount (%)", 1.0, 15.0, 5.0) / 100
            mean_growth = st.slider("Mean Terminal Growth (%)", 0.0, 8.0, 3.0) / 100
            std_growth = st.slider("Std Dev Growth (%)", 0.5, 5.0, 1.5) / 100
            if st.button("Run Simulation"):
                evs = []
                fcff_list = [-450, -250, 100, 900, 1925]
                for _ in range(int(simulations)):
                    r = np.random.normal(mean_discount, std_discount)
                    g = np.random.normal(mean_growth, std_growth)
                    if r <= g or r < 0.1: continue
                    pv_explicit = sum(fcff_list[i] / (1 + r)**(i+1) for i in range(5))
                    tv = fcff_list[4] * (1 + g) / (r - g)
                    pv_tv = tv / (1 + r)**5
                    evs.append(pv_explicit + pv_tv)
                if evs:
                    mean_ev = np.mean(evs)
                    p10 = np.percentile(evs, 10)
                    p90 = np.percentile(evs, 90)
                    st.metric("Mean EV", f"${mean_ev:,.0f}k")
                    st.metric("P10 / P90", f"${p10:,.0f}k / ${p90:,.0f}k")
                    chart = alt.Chart(pd.DataFrame({"EV": evs})).mark_bar().encode(alt.X("EV", bin=True), y='count()')
                    st.altair_chart(chart, use_container_width=True)
                    save_data = {"mean_ev": mean_ev, "p10": p10, "p90": p90}
                    save_analysis("Monte Carlo", save_data)
                    st.success("Monte Carlo simulation saved.")

        # Scenario Planning
        elif sidebar_selection == "Scenario Planning":
            st.header("🔮 Scenario Planning")
            st.markdown(f"**Deal:** {current_deal.company_name}")
            scenarios = ["Base Case", "Optimistic", "Pessimistic"]
            scenario_inputs = {}
            for scenario in scenarios:
                with st.expander(f"{scenario} Assumptions"):
                    col1, col2 = st.columns(2)
                    with col1:
                        scenario_inputs[scenario] = {}
                        scenario_inputs[scenario]["mrr"] = st.number_input(f"Starting MRR {scenario}", value=10000 if "Base" in scenario else 15000 if "Optimistic" in scenario else 5000, key=f"mrr_{scenario}")
                        scenario_inputs[scenario]["growth"] = st.slider(f"Growth % {scenario}", 5.0, 30.0, 15.0 if "Base" in scenario else 25.0 if "Optimistic" in scenario else 5.0, key=f"growth_{scenario}") / 100
                    with col2:
                        scenario_inputs[scenario]["churn"] = st.slider(f"Churn % {scenario}", 1.0, 10.0, 4.0 if "Base" in scenario else 2.0 if "Optimistic" in scenario else 8.0, key=f"churn_{scenario}") / 100
            if st.button("Generate Scenarios"):
                months = 36
                scenario_dfs = {}
                for scenario in scenarios:
                    mrr = [scenario_inputs[scenario]["mrr"]]
                    for _ in range(months - 1):
                        mrr.append(mrr[-1] * (1 + scenario_inputs[scenario]["growth"]) * (1 - scenario_inputs[scenario]["churn"]))
                    df = pd.DataFrame({"Month": range(1, months+1), "MRR": mrr, "Scenario": scenario})
                    scenario_dfs[scenario] = df
                combined = pd.concat(scenario_dfs.values())
                chart = alt.Chart(combined).mark_line().encode(x='Month', y='MRR', color='Scenario')
                st.altair_chart(chart, use_container_width=True)
                save_data = {s: scenario_dfs[s].to_dict() for s in scenarios}
                save_analysis("Scenario Planning", save_data)
                st.success("Scenario planning saved.")

        # Generate Report
        elif sidebar_selection == "Generate Report":
            st.header("📑 Generate Comprehensive Report")
            st.markdown(f"**Deal:** {current_deal.company_name}")

            if st.button("Generate AI Investment Memo"):
                all_data = f"Deal: {current_deal.company_name}\nStage: {current_deal.stage}\nNotes: {current_deal.notes or 'None'}\n\n"
                for a in session_db.query(Analysis).filter_by(deal_id=current_deal.id).all():
                    all_data += f"{a.section}:\n{a.data}\n\n"
                if openai_client:
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a professional VC investment analyst. Write a concise investment memo including executive summary, key risks, strengths, and recommendation."},
                                {"role": "user", "content": all_data}
                            ],
                            max_tokens=2000
                        )
                        st.markdown("### AI Investment Memo")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI memo generation failed: {e}")
                else:
                    st.info("Configure OpenAI API key for AI memo generation.")

            if st.button("Generate PDF Report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, f"Due Diligence Report - {current_deal.company_name}", ln=1, align='C')
                pdf.set_font("Arial", size=12)
                pdf.ln(10)
                pdf.cell(0, 10, f"Prepared by: {name} | Date: {datetime.date.today()} | Stage: {current_deal.stage}", ln=1)
                pdf.ln(10)
                pdf.cell(0, 10, "Summary of Analyses:", ln=1)
                analyses = session_db.query(Analysis).filter_by(deal_id=current_deal.id).order_by(Analysis.timestamp.desc()).all()
                for a in analyses:
                    data = json.loads(a.data) if a.data else {}
                    score = data.get("score", "N/A")
                    pdf.cell(0, 10, f"- {a.section} ({a.timestamp.date()}): Score {score}", ln=1)
                buffer = io.BytesIO()
                pdf.output(buffer)
                buffer.seek(0)
                st.download_button("Download Report", buffer, f"DD_Report_{current_deal.company_name}.pdf", "application/pdf")

            if st.button("Export All Deal Data to Excel"):
                all_data = {}
                for a in session_db.query(Analysis).filter_by(deal_id=current_deal.id).all():
                    try:
                        section_data = json.loads(a.data)
                    except:
                        section_data = {"raw_data": a.data}
                    all_data[a.section] = section_data
                df = pd.json_normalize(all_data)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Deal_Data')
                buffer.seek(0)
                st.download_button("Download Full Deal Export", buffer, f"{current_deal.company_name}_full_export.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    session_db.close()

elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")