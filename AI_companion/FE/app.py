import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# ---------------------------
# Page Configuration & Styling
# ---------------------------
st.set_page_config(
    page_title="AI Lab Compliance Copilot",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .stApp, body { background-color: #f3f4f6; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

        div[data-testid="stChatMessage"] {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 16px !important;
            padding: 12px 14px !important;
            margin-bottom: 10px !important;
        }
        div[data-testid="stChatMessage"] p {
            color: #111827 !important;
            font-size: 1rem !important;
            line-height: 1.45 !important;
        }

        div[data-testid="stChatMessage"] > div:first-child {
            background-color: #dbeafe !important;
            border-radius: 50% !important;
            border: 2px solid #3b82f6 !important;
            padding: 2px !important;
        }

        div[data-testid="stChatMessage"][data-kind="user"] > div:first-child {
            background-color: #dbeafe !important;
            border: 2px solid #3b82f6 !important;
        }

        div[data-testid="stChatMessage"][data-kind="assistant"] > div:first-child {
            background-color: #dcfce7 !important;
            border: 2px solid #22c55e !important;
        }

        .stTextArea textarea {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 12px !important;
            color: #111827 !important;
        }

        button[kind="primary"] {
            background-color: #2563eb !important;
            color: white !important;
            border-radius: 10px !important;
            height: 42px !important;
            padding: 0 18px !important;
            font-weight: 600 !important;
            border: 1px solid #1d4ed8 !important;
        }

        h1, h2, h3, h4 { color: #111827 !important; }

        .card {
            border: 1px solid #e5e7eb;
            background: #ffffff;
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }
        
        .deviation-card {
            border: 1px solid #e5e7eb;
            background: #ffffff;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .critical { border-left: 4px solid #ef4444; }
        .major { border-left: 4px solid #f59e0b; }
        .minor { border-left: 4px solid #10b981; }
        
        .download-btn {
            background-color: #10b981 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 6px 12px !important;
            font-size: 0.8rem !important;
        }
        
        .sample-question {
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            margin: 4px 0 !important;
            cursor: pointer !important;
            transition: all 0.2s !important;
        }
        
        .sample-question:hover {
            background: #e2e8f0 !important;
            border-color: #cbd5e1 !important;
        }
        
        .sop-reference {
            font-size: 0.8rem;
            color: #6b7280;
            font-style: italic;
            margin-top: 8px;
            padding: 4px 8px;
            background: #f3f4f6;
            border-radius: 4px;
        }
        
        .alert-critical {
            background: #fef2f2 !important;
            border: 1px solid #fecaca !important;
            border-left: 4px solid #ef4444 !important;
        }
        
        .alert-warning {
            background: #fffbeb !important;
            border: 1px solid #fed7aa !important;
            border-left: 4px solid #f59e0b !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Backend URL Configuration
# ---------------------------
BACKEND_URL = "http://localhost:8000"

# ---------------------------
# Session State Initialization
# ---------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "clear_input" not in st.session_state:
    st.session_state["clear_input"] = False

if "deviation_reports" not in st.session_state:
    st.session_state["deviation_reports"] = []

if "trends_data" not in st.session_state:
    st.session_state["trends_data"] = None

if "available_sops" not in st.session_state:
    st.session_state["available_sops"] = []

if "current_query" not in st.session_state:
    st.session_state["current_query"] = ""

if "query_submitted" not in st.session_state:
    st.session_state["query_submitted"] = False

if "critical_deviations" not in st.session_state:
    st.session_state["critical_deviations"] = None

if "compliance_trends" not in st.session_state:
    st.session_state["compliance_trends"] = None

if "compliance_dashboard" not in st.session_state:
    st.session_state["compliance_dashboard"] = None

if "recent_alerts" not in st.session_state:
    st.session_state["recent_alerts"] = None

# ---------------------------
# Utility Functions
# ---------------------------
def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def upload_sop_file(file):
    """Upload SOP file to backend"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{BACKEND_URL}/upload-sop", files=files)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def list_sops():
    """Get list of available SOPs from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/list-sops")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def process_sops():
    """Process SOPs in backend"""
    try:
        response = requests.post(f"{BACKEND_URL}/process")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def process_deviation_samples():
    """Process deviation samples in backend"""
    try:
        response = requests.post(f"{BACKEND_URL}/process-deviations")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def query_sop(query):
    """Query SOP system"""
    try:
        response = requests.post(f"{BACKEND_URL}/query", data={"query": query})
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def process_incident(incident_description, generate_pdf=True):
    """Report incident to backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/report-incident",
            data={"incident": incident_description, "generate_pdf": generate_pdf}
        )
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def get_deviation_trends(days=30):
    """Get deviation trends from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/deviation-trends?days={days}")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def get_retraining_suggestions():
    """Get retraining suggestions from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/retraining-suggestions")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def create_structured_deviation(incident_description, severity, category, generate_pdf=True):
    """Create structured deviation report"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/deviation-prompt",
            data={
                "incident_description": incident_description,
                "severity": severity,
                "category": category,
                "generate_pdf": generate_pdf
            }
        )
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def list_reports():
    """List all generated reports"""
    try:
        response = requests.get(f"{BACKEND_URL}/list-reports")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def get_critical_deviations():
    """Get flagged critical deviations"""
    try:
        response = requests.get(f"{BACKEND_URL}/flag-critical-deviations")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def get_compliance_trends():
    """Get non-compliance trends"""
    try:
        response = requests.get(f"{BACKEND_URL}/compliance-trends")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def get_compliance_dashboard():
    """Get comprehensive compliance dashboard"""
    try:
        response = requests.get(f"{BACKEND_URL}/compliance-dashboard")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def get_real_time_alerts():
    """Get real-time compliance alerts"""
    try:
        response = requests.get(f"{BACKEND_URL}/real-time-alerts")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, f"Connection error: {e}"

def set_sample_question(question):
    """Set sample question in query input"""
    st.session_state["current_query"] = question
    st.session_state["query_submitted"] = False

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("⚙️ Data & Settings")
    
    # Backend Health Check
    backend_healthy, health_data = check_backend_health()
    if backend_healthy:
        if health_data:
            st.caption(f"SOP Files: {health_data.get('sop_files_available', 0)}")
            st.caption(f"Deviation Samples: {health_data.get('deviation_samples_available', 0)}")
    else:
        st.error("❌ Backend Not Connected")
        st.caption("Make sure backend is running on localhost:8000")
    
    # Real-time Alerts Section
    st.markdown("---")
    st.subheader("🚨 Real-time Alerts")
    
    if st.button("🔄 Check Alerts", use_container_width=True, key="check_alerts_btn"):
        with st.spinner("Checking for critical deviations..."):
            success, alerts = get_real_time_alerts()
            if success and alerts.get('alerts_count', 0) > 0:
                st.session_state["recent_alerts"] = alerts
                st.warning(f"🚨 {alerts['alerts_count']} critical alerts found!")
            else:
                st.session_state["recent_alerts"] = None
                st.success("✅ No critical alerts")
    
    if "recent_alerts" in st.session_state and st.session_state["recent_alerts"]:
        alerts_data = st.session_state["recent_alerts"]
        for alert in alerts_data.get('alerts', [])[:3]:  # Show top 3
            st.error(f"**{alert['title']}**")
            st.caption(alert['description'][:100] + "...")
    
    st.markdown("---")
    
    # SOP Management
    st.subheader("📚 SOP Management")
    
    # List available SOPs
    if st.button("🔄 Refresh SOP List", use_container_width=True, key="refresh_sops_btn"):
        success, sops_data = list_sops()
        if success:
            st.session_state["available_sops"] = sops_data.get("sops", [])
            st.success(f"Found {len(st.session_state['available_sops'])} SOPs")
        else:
            st.error("Failed to load SOPs")
    
    if st.session_state["available_sops"]:
        st.write("**Available SOPs:**")
        for sop in st.session_state["available_sops"][:5]:  # Show first 5
            st.caption(f"• {sop}")
    
    # SOP Upload
    uploaded_pdf = st.file_uploader("📄 Upload New SOP PDF", type=["pdf"], key="pdf_up")
    if uploaded_pdf is not None:
        with st.spinner(f"Uploading {uploaded_pdf.name}..."):
            success, result = upload_sop_file(uploaded_pdf)
            if success:
                st.success(f"✅ {uploaded_pdf.name} uploaded successfully!")
                # Refresh SOP list
                success, sops_data = list_sops()
                if success:
                    st.session_state["available_sops"] = sops_data.get("sops", [])
            else:
                st.error(f"❌ Upload failed: {result}")
    
    # SOP Processing
    if st.button("📥 Process SOPs", use_container_width=True, key="process_sops_btn"):
        with st.spinner("Embedding and processing SOP documents..."):
            success, result = process_sops()
            if success:
                st.success("✅ SOPs processed successfully!")
                # Refresh SOP list
                success, sops_data = list_sops()
                if success:
                    st.session_state["available_sops"] = sops_data.get("sops", [])
            else:
                st.error(f"❌ Failed to process SOPs: {result}")
    
    st.markdown("---")
    
    # Deviation Analysis
    st.subheader("📊 Deviation Analysis")
    
    if st.button("🔄 Process Deviation Samples", use_container_width=True, key="process_deviations_btn"):
        with st.spinner("Processing deviation samples for trend analysis..."):
            success, result = process_deviation_samples()
            if success:
                st.success("✅ Deviation samples processed successfully!")
            else:
                st.error(f"❌ Failed to process deviations: {result}")
    
    # Enhanced Compliance Analysis Section
    st.subheader("🚨 Compliance Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚩 Flag Critical", use_container_width=True, key="flag_deviations_btn"):
            with st.spinner("Analyzing for critical deviations..."):
                success, critical_data = get_critical_deviations()
                if success:
                    st.session_state["critical_deviations"] = critical_data
                    flagged_count = critical_data.get('flagged_count', 0)
                    if flagged_count > 0:
                        st.warning(f"🚩 {flagged_count} critical deviations!")
                    else:
                        st.success("✅ No critical issues")
                else:
                    st.error("Failed to flag deviations")
    
    with col2:
        if st.button("📈 Find Trends", use_container_width=True, key="find_trends_btn"):
            with st.spinner("Analyzing non-compliance patterns..."):
                success, trends_data = get_compliance_trends()
                if success:
                    st.session_state["compliance_trends"] = trends_data
                    trends_count = trends_data.get('trends_identified', 0)
                    if trends_count > 0:
                        st.info(f"📈 {trends_count} trends found")
                    else:
                        st.success("✅ No significant trends")
                else:
                    st.error("Failed to analyze trends")
    
    if st.button("📋 Generate Dashboard", use_container_width=True, key="generate_dashboard_btn"):
        with st.spinner("Generating compliance dashboard..."):
            success, dashboard_data = get_compliance_dashboard()
            if success:
                st.session_state["compliance_dashboard"] = dashboard_data
                st.success("✅ Dashboard generated!")
            else:
                st.error("Failed to generate dashboard")
    
    # Quick Status Display
    if "critical_deviations" in st.session_state and st.session_state["critical_deviations"]:
        critical_data = st.session_state["critical_deviations"]
        flagged_count = critical_data.get('flagged_count', 0)
        if flagged_count > 0:
            st.error(f"**Active Critical Issues:** {flagged_count}")
    
    if "compliance_trends" in st.session_state and st.session_state["compliance_trends"]:
        trends_data = st.session_state["compliance_trends"]
        trends_count = trends_data.get('trends_identified', 0)
        if trends_count > 0:
            st.warning(f"**Active Trends:** {trends_count}")
    
    st.markdown("---")
    
    # Report Management
    st.subheader("📋 Reports")
    if st.button("🔄 Refresh Reports", use_container_width=True, key="refresh_reports_btn"):
        success, reports = list_reports()
        if success:
            st.session_state["available_reports"] = reports
            st.success("Reports refreshed!")
        else:
            st.error("Failed to load reports")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Quick Scan", use_container_width=True, key="quick_scan_btn"):
            with st.spinner("Quick scanning for issues..."):
                success, critical_data = get_critical_deviations()
                if success:
                    flagged_count = critical_data.get('flagged_count', 0)
                    if flagged_count > 0:
                        st.error(f"🚨 {flagged_count} critical issues!")
                    else:
                        st.success("✅ System clean")
    
    with col2:
        if st.button("📊 Trends Only", use_container_width=True, key="trends_only_btn"):
            with st.spinner("Checking trends..."):
                success, trends_data = get_compliance_trends()
                if success:
                    trends_count = trends_data.get('trends_identified', 0)
                    if trends_count > 0:
                        st.warning(f"📈 {trends_count} patterns found")
                    else:
                        st.success("✅ No trends")
    
    # System Status
    st.markdown("---")
    st.subheader("🔍 System Status")
    
    # Show current session state status
    status_items = []
    
    if st.session_state.get("available_sops"):
        status_items.append(f"📚 SOPs: {len(st.session_state['available_sops'])}")
    
    if st.session_state.get("deviation_reports"):
        status_items.append(f"⚠️ Reports: {len(st.session_state['deviation_reports'])}")
    
    if st.session_state.get("critical_deviations"):
        critical_count = st.session_state["critical_deviations"].get('flagged_count', 0)
        if critical_count > 0:
            status_items.append(f"🚨 Critical: {critical_count}")
    
    if st.session_state.get("compliance_trends"):
        trends_count = st.session_state["compliance_trends"].get('trends_identified', 0)
        if trends_count > 0:
            status_items.append(f"📈 Trends: {trends_count}")
    
    if status_items:
        st.write("**Current Status:**")
        for item in status_items:
            st.caption(item)
    else:
        st.caption("No active analysis data")
    
    # Clear All Data
    st.markdown("---")
    if st.button("🗑️ Clear All Data", use_container_width=True, key="clear_all_btn"):
        st.session_state["history"] = []
        st.session_state["deviation_reports"] = []
        st.session_state["critical_deviations"] = None
        st.session_state["compliance_trends"] = None
        st.session_state["compliance_dashboard"] = None
        st.session_state["recent_alerts"] = None
        st.session_state["current_query"] = ""
        st.session_state["query_submitted"] = False
        st.success("All session data cleared!")
        st.rerun()

# ---------------------------
# Main Layout
# ---------------------------
st.title("🧪 AI Lab Compliance Copilot")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["💬 SOP Chat", "⚠️ Report Incident", "📈 Trends & Training", "📊 Compliance Dashboard"])

# ---------------------------
# Tab 1: SOP Chat Interface
# ---------------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask about SOPs")
        
        # Display chat history
        for turn in st.session_state["history"]:
            with st.chat_message("user"):
                st.write(turn["q"])
            with st.chat_message("assistant"):
                st.write(turn["a"])
                # Show SOP references as subtle caption
                if turn.get("references"):
                    st.markdown(f'<div class="sop-reference">Based on: {", ".join(turn["references"])}</div>', unsafe_allow_html=True)
        
        # Handle query submission from sample questions
        if st.session_state.get("query_submitted", False):
            query = st.session_state["current_query"]
            if query.strip():
                with st.spinner("Searching SOP documents..."):
                    success, result = query_sop(query)
                    
                    if success:
                        answer_text = result.get("answer", "No answer found.")
                        references = result.get("sop_references", [])
                        
                        # Add to history
                        st.session_state["history"].append({
                            "q": query.strip(), 
                            "a": answer_text,
                            "references": references
                        })
                        
                        # Reset the flag and clear query
                        st.session_state["query_submitted"] = False
                        st.session_state["current_query"] = ""
                        st.rerun()
                    else:
                        st.error(f"Query failed: {result}")
                        st.session_state["query_submitted"] = False
        
        # Chat input - use a different key for the widget
        query_input = st.text_area(
            "Ask a question about SOPs, procedures, or compliance...", 
            height=100, 
            label_visibility="collapsed", 
            key="query_input_widget",
            value=st.session_state["current_query"],
            placeholder="e.g., What is the temperature limit for FBD? What are the cleaning procedures for compression machines?"
        )
        
        col_btn1, col_btn2 = st.columns([1, 6])
        with col_btn1:
            ask_btn = st.button("Submit", type="primary", use_container_width=True, key="submit_query_btn")
        with col_btn2:
            clear_btn = st.button("Clear Chat", use_container_width=True, key="clear_chat_btn")
        
        if clear_btn:
            st.session_state["history"] = []
            st.session_state["current_query"] = ""
            st.session_state["query_submitted"] = False
            st.rerun()
        
        if ask_btn and query_input.strip():
            st.session_state["current_query"] = query_input
            st.session_state["query_submitted"] = True
            st.rerun()
    
    with col2:
        st.subheader("Quick Actions")
        
        st.markdown("---")
        st.subheader("📋 Sample Questions")
        
        sample_questions = [
            "What is the compression machine pressure limit?",
            "How to clean a vibro sifter?",
            "What are the environmental monitoring requirements?",
            "Procedure for handling deviations",
            "Temperature limits for fluid bed dryer",
            "Documentation requirements for batch manufacturing",
            "Sampling frequency for in-process checks",
            "Calibration procedure for balances",
            "Cleaning validation requirements",
            "Personnel training requirements"
        ]
        
        for question in sample_questions:
            if st.button(
                question, 
                use_container_width=True, 
                key=f"sample_{hash(question)}"
            ):
                set_sample_question(question)
                st.session_state["query_submitted"] = True
                st.rerun()

# ---------------------------
# Tab 2: Incident Reporting
# ---------------------------
with tab2:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Report New Incident")
            
        with st.form("structured_incident_form"):
            incident_desc = st.text_area("Incident Description:", 
                                        key="structured_incident",
                                        placeholder="Brief description of what happened")
            
            col1, col2 = st.columns(2)
            with col1:
                severity = st.selectbox("Severity Level:", 
                                        ["minor", "major", "critical"],
                                        key="severity_select")
                category = st.selectbox("Category:",
                                        ["equipment", "process", "documentation", "training", "environmental"],
                                        key="category_select")
            
            with col2:
                department = st.selectbox("Department:",
                                        ["Manufacturing", "Quality Control", "Packaging", "Maintenance"],
                                        key="dept_select")
                generate_pdf = st.checkbox("Generate PDF Report", value=True, key="structured_pdf")
            
            submitted = st.form_submit_button("🚨 Create Deviation Report", type="primary", key="create_deviation_btn")
            
            if submitted and incident_desc.strip():
                with st.spinner("Creating structured deviation report..."):
                    success, result = create_structured_deviation(incident_desc, severity, category, generate_pdf)
                    
                    if success:
                        st.success("✅ Deviation report created successfully!")
                        if "report" in result:
                            st.session_state["deviation_reports"].append(result)
                            
                            # Show download button for PDF
                            report_data = result["report"]
                            if report_data.get("pdf_report_path"):
                                pdf_filename = os.path.basename(report_data["pdf_report_path"])
                                st.download_button(
                                    label="📄 Download PDF Report",
                                    data=requests.get(f"{BACKEND_URL}/download-pdf/{pdf_filename}").content,
                                    file_name=pdf_filename,
                                    mime="application/pdf",
                                    use_container_width=True,
                                    key=f"dl_structured_{pdf_filename}"
                                )

    with col2:
        st.subheader("Recent Deviation Reports")
        
        if st.session_state["deviation_reports"]:
            for i, report in enumerate(st.session_state["deviation_reports"][-5:]):
                deviation_data = report.get("deviation_analysis", {})
                severity = deviation_data.get("severity_level", "unknown")
                incident_preview = report.get('incident', 'Unknown incident')[:80] + "..." if len(report.get('incident', '')) > 80 else report.get('incident', 'Unknown incident')
                
                st.markdown(f"""
                <div class="deviation-card {severity}">
                    <strong>{incident_preview}</strong><br>
                    <small>Severity: {severity.upper()} | Category: {deviation_data.get('deviation_category', 'unknown')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No deviation reports yet. Report an incident to see them here.")

# ---------------------------
# Tab 3: Trends & Training
# ---------------------------
with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Deviation Trends & Flagging")
        
        # Quick flagging section
        st.write("**🚨 Quick Flagging**")
        
        if st.button("🔍 Scan for Critical Issues", use_container_width=True, key="scan_critical_btn"):
            with st.spinner("Scanning for critical deviations..."):
                success, critical_data = get_critical_deviations()
                if success:
                    flagged_count = critical_data.get('flagged_count', 0)
                    if flagged_count > 0:
                        st.error(f"🚨 {flagged_count} critical deviations found!")
                        for deviation in critical_data.get('critical_deviations', [])[:3]:
                            st.write(f"• {deviation.get('content', '')[:100]}...")
                    else:
                        st.success("✅ No critical issues found")
        
        # Display quick trends if available
        if "compliance_trends_quick" in st.session_state:
            trends_data = st.session_state["compliance_trends_quick"]
            st.write("**📊 Quick Trends Analysis**")
            for trend in trends_data.get('compliance_trends', [])[:2]:
                st.write(f"• {trend.get('pattern', '')}: {trend.get('analysis', {}).get('severity', '')}")
        
        # Historical Trends Analysis
        st.markdown("---")
        st.subheader("📊 Historical Trends")
        
        if st.button("📈 Analyze Historical Trends", use_container_width=True, key="analyze_historical_btn"):
            with st.spinner("Analyzing historical deviation patterns..."):
                success, trends = get_deviation_trends(30)
                if success:
                    st.session_state["trends_data"] = trends
                    st.success("Trends analysis completed!")
                else:
                    st.error("Failed to analyze trends")
        
        if st.session_state["trends_data"]:
            trends = st.session_state["trends_data"]
            if "error" not in trends:
                st.success(f"✅ Analyzed {trends.get('reports_analyzed', 0)} reports")
                
                # Display trends analysis
                st.write("**Trends Summary:**")
                st.write(trends.get("trends_analysis", "No analysis available"))
            else:
                st.error(trends["error"])
    
    with col2:
        st.subheader("🎓 Retraining & SOP Reinforcement")
        
        if st.button("🔄 Generate Training Suggestions", use_container_width=True, key="generate_training_btn"):
            with st.spinner("Analyzing deviations and generating training suggestions..."):
                success, suggestions = get_retraining_suggestions()
                
                if success and "error" not in suggestions:
                    st.success("✅ Training suggestions generated!")
                    
                    # Display suggestions
                    st.write("**Comprehensive Training Program:**")
                    st.write(suggestions.get("suggestions", "No specific suggestions available"))
                    
                    # Show program details
                    if suggestions.get("program_id"):
                        st.caption(f"Program ID: {suggestions['program_id']}")
                else:
                    st.error("Failed to generate training suggestions")
        
        st.markdown("---")
        st.subheader("📋 Common Training Needs")
        
        training_needs = [
            "Good Documentation Practices (GDP)",
            "Equipment Operation & Maintenance",
            "Environmental Monitoring Procedures",
            "Deviation Reporting & Investigation",
            "Root Cause Analysis Techniques",
            "GMP Compliance Awareness",
            "Quality Control Testing Methods",
            "Batch Record Documentation"
        ]
        
        for need in training_needs:
            st.write(f"• {need}")

# ---------------------------
# Tab 4: Compliance Dashboard
# ---------------------------
with tab4:
    st.subheader("📊 Real-time Compliance Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Dashboard", use_container_width=True, key="refresh_dashboard_btn"):
            with st.spinner("Updating dashboard..."):
                success, dashboard_data = get_compliance_dashboard()
                if success:
                    st.session_state["compliance_dashboard"] = dashboard_data
                    st.rerun()
    
    # Display Dashboard Data
    # Display Dashboard Data
if "compliance_dashboard" in st.session_state and st.session_state["compliance_dashboard"] is not None:
    dashboard_data = st.session_state["compliance_dashboard"]
    
    # Check if we have a valid dashboard response
    if dashboard_data and "dashboard" in dashboard_data:
        dashboard = dashboard_data.get("dashboard", {})
        metrics = dashboard.get("metrics", {})
        
        # Critical Deviations Section
        st.markdown("---")
        st.subheader("🚨 Critical Deviations Flagged")
        
        critical_deviations = dashboard.get("critical_deviations", [])
        if critical_deviations:
            for i, deviation in enumerate(critical_deviations[:5]):  # Show top 5
                analysis = deviation.get('analysis', {})
                with st.expander(f"🚩 Critical Deviation {i+1} - {analysis.get('risk_level', 'Unknown').upper()}"):
                    st.write(f"**Source:** {deviation.get('source_file', 'Unknown')}")
                    st.write(f"**Content:** {deviation.get('content', 'No content')}")
                    st.write(f"**Risk Level:** {analysis.get('risk_level', 'Unknown')}")
                    st.write(f"**Affected Areas:** {', '.join(analysis.get('affected_areas', []))}")
                    st.write("**Recommended Actions:**")
                    for action in analysis.get('recommended_actions', []):
                        st.write(f"• {action}")
        else:
            st.success("✅ No critical deviations flagged")
        
        # Compliance Trends Section
        st.markdown("---")
        st.subheader("📈 Non-Compliance Trends")
        
        compliance_trends = dashboard.get("compliance_trends", [])
        if compliance_trends:
            for i, trend in enumerate(compliance_trends[:3]):  # Show top 3
                analysis = trend.get('analysis', {})
                with st.expander(f"📊 Trend {i+1}: {analysis.get('trend_type', 'Unknown').title()}"):
                    st.write(f"**Pattern:** {trend.get('pattern', 'Unknown')}")
                    st.write(f"**Severity:** {analysis.get('severity', 'Unknown')}")
                    st.write(f"**Recurrence:** {analysis.get('recurrence_frequency', 'Unknown')}")
                    st.write(f"**Root Cause:** {analysis.get('root_cause_pattern', 'Unknown')}")
                    st.write(f"**Departments Affected:** {', '.join(analysis.get('departments_affected', []))}")
                    st.write("**Preventive Measures:**")
                    for measure in analysis.get('preventive_measures', []):
                        st.write(f"• {measure}")
        else:
            st.info("No significant compliance trends identified")
        
        # Recommendations Section
        st.markdown("---")
        st.subheader("🎯 Actionable Recommendations")
        
        recommendations = dashboard.get("recommendations", {})
        if recommendations:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🚀 Immediate Actions**")
                for action in recommendations.get('immediate_actions', []):
                    st.write(f"• {action}")
                
                st.write("**🎓 Training Priorities**")
                for priority in recommendations.get('training_priorities', []):
                    st.write(f"• {priority}")
            
            with col2:
                st.write("**🛡️ Preventive Measures**")
                for measure in recommendations.get('preventive_measures', []):
                    st.write(f"• {measure}")
                
                st.write("**⚙️ System Improvements**")
                for improvement in recommendations.get('system_improvements', []):
                    st.write(f"• {improvement}")
        else:
            st.info("No specific recommendations available")
    
    else:
        st.error("❌ Failed to generate dashboard data")
        if dashboard_data and "error" in dashboard_data:
            st.error(f"Error: {dashboard_data['error']}")
    # else:
    #     st.info("Click 'Generate Dashboard' to see compliance analytics")
        
    #     # Quick actions
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         if st.button("🚩 Check Critical Deviations", use_container_width=True, key="check_critical_btn"):
    #             with st.spinner("Checking for critical issues..."):
    #                 success, critical_data = get_critical_deviations()
    #                 if success:
    #                     st.session_state["critical_deviations_quick"] = critical_data
    #                     st.rerun()
        
    #     with col2:
    #         if st.button("📈 Quick Trends Analysis", use_container_width=True, key="quick_trends_btn"):
    #             with st.spinner("Analyzing trends..."):
    #                 success, trends_data = get_compliance_trends()
    #                 if success:
    #                     st.session_state["compliance_trends_quick"] = trends_data
    #                     st.rerun()

