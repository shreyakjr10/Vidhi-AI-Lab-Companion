import os
import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm
import requests
import redis
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import glob
import shutil

# ========== CONFIG ==========
SOP_FOLDER = "BE/sop_docs"
DEVIATION_FOLDER = "deviation_reports"
DEVIATION_SAMPLE_FOLDER = "deviation_samples"
TRAINING_FOLDER = "training_recommendations"
PDF_REPORTS_FOLDER = "pdf_reports"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
GROQ_API_KEY = "your_grok_api_key"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 3

# ========== INIT ==========
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create folders
for folder in [SOP_FOLDER, DEVIATION_FOLDER, DEVIATION_SAMPLE_FOLDER, TRAINING_FOLDER, PDF_REPORTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ========== PDF REPORT GENERATOR ==========
class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for pharmaceutical reports"""
        self.styles.add(ParagraphStyle(
            name='PharmaTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.darkblue,
            spaceAfter=12,
            alignment=1
        ))
        
        self.styles.add(ParagraphStyle(
            name='PharmaHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.darkred,
            spaceAfter=6,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='PharmaBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leading=12
        ))

    def create_deviation_report_pdf(self, deviation_data, query, contexts, deviation_id):
        """Create comprehensive PDF deviation report"""
        filename = f"{deviation_id}_REPORT.pdf"
        filepath = os.path.join(PDF_REPORTS_FOLDER, filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=A4, topMargin=0.5*inch)
        story = []
        
        # Header
        story.append(self._create_header(deviation_id))
        story.append(Spacer(1, 0.2*inch))
        
        # Executive Summary
        story.append(self._create_executive_summary(deviation_data, query))
        story.append(Spacer(1, 0.1*inch))
        
        # Deviation Details
        story.append(self._create_deviation_details(deviation_data))
        story.append(Spacer(1, 0.1*inch))
        
        # Risk Assessment
        story.append(self._create_risk_assessment(deviation_data))
        story.append(Spacer(1, 0.1*inch))
        
        # Immediate Actions
        story.append(self._create_immediate_actions(deviation_data))
        story.append(Spacer(1, 0.1*inch))
        
        # Investigation Requirements
        story.append(self._create_investigation_requirements(deviation_data))
        story.append(Spacer(1, 0.1*inch))
        
        # Training Recommendations
        if deviation_data.get("training_implications", {}).get("needs_retraining", False):
            story.append(self._create_training_recommendations(deviation_data))
            story.append(Spacer(1, 0.1*inch))
        
        # SOP References
        story.append(self._create_sop_references(contexts))
        story.append(Spacer(1, 0.1*inch))
        
        # Footer
        story.append(self._create_footer())
        
        try:
            doc.build(story)
            print(f"📄 PDF Report Generated: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
            return None

    def _create_header(self, deviation_id):
        """Create report header"""
        header_elements = []
        
        title = Paragraph("PHARMACEUTICAL DEVIATION REPORT", self.styles['PharmaTitle'])
        header_elements.append(title)
        header_elements.append(Spacer(1, 0.1*inch))
        
        deviation_info = [
            ["Deviation ID:", deviation_id],
            ["Date Generated:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ["Report Type:", "Comprehensive Deviation Analysis"],
            ["Regulatory Reference:", "FDA 21 CFR 211.100, EU GMP Chapter 1"]
        ]
        
        table = Table(deviation_info, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        header_elements.append(table)
        return header_elements

    def _create_executive_summary(self, deviation_data, query):
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.styles['PharmaHeading']))
        
        severity = deviation_data.get('severity_level', 'Unknown').upper()
        category = deviation_data.get('deviation_category', 'Unknown').replace('_', ' ').title()
        
        summary_text = f"""
        <b>Incident Overview:</b> {query}<br/>
        <b>Severity Level:</b> {severity}<br/>
        <b>Category:</b> {category}<br/>
        <b>Confidence Score:</b> {deviation_data.get('confidence_score', 0) * 100:.1f}%<br/>
        <b>Immediate Impact:</b> Requires immediate investigation and corrective actions.<br/>
        <b>Regulatory Significance:</b> This deviation impacts GMP compliance and requires thorough documentation.
        """
        
        elements.append(Paragraph(summary_text, self.styles['PharmaBody']))
        return elements

    def _create_deviation_details(self, deviation_data):
        """Create deviation details section"""
        elements = []
        
        elements.append(Paragraph("DEVIATION CLASSIFICATION", self.styles['PharmaHeading']))
        
        details = [
            ["Parameter", "Value", "Risk Level"],
            ["Deviation Type", deviation_data.get('deviation_type', 'Unknown').title(), self._get_risk_style(deviation_data.get('severity_level'))],
            ["Severity Level", deviation_data.get('severity_level', 'Unknown').title(), self._get_risk_style(deviation_data.get('severity_level'))],
            ["Category", deviation_data.get('deviation_category', 'Unknown').replace('_', ' ').title(), "Medium"],
            ["Detection Method", "Automated AI Compliance Monitoring", "Low"]
        ]
        
        table = Table(details, colWidths=[1.5*inch, 3*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white)
        ]))
        
        elements.append(table)
        return elements

    def _create_risk_assessment(self, deviation_data):
        """Create risk assessment section"""
        elements = []
        
        elements.append(Paragraph("RISK ASSESSMENT MATRIX", self.styles['PharmaHeading']))
        
        risk_data = deviation_data.get('risk_assessment', {})
        risk_matrix = [
            ["Risk Category", "Impact Level", "Description"],
            ["Product Quality", risk_data.get('product_quality_impact', 'Unknown').title(), 
             "Impact on product specifications and quality attributes"],
            ["Patient Safety", risk_data.get('patient_safety_impact', 'Unknown').title(), 
             "Potential impact on patient health and safety"],
            ["Regulatory Compliance", risk_data.get('regulatory_impact', 'Unknown').title(), 
             "Compliance with FDA, EMA, and other regulatory requirements"],
            ["Business Impact", risk_data.get('business_impact', 'Unknown').title(), 
             "Financial and operational consequences"]
        ]
        
        table = Table(risk_matrix, colWidths=[1.5*inch, 1.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white)
        ]))
        
        elements.append(table)
        return elements

    def _create_immediate_actions(self, deviation_data):
        """Create immediate actions section"""
        elements = []
        
        elements.append(Paragraph("IMMEDIATE CORRECTIVE ACTIONS", self.styles['PharmaHeading']))
        
        actions = deviation_data.get('immediate_actions', [])
        if not actions:
            actions = [
                "Stop affected process immediately",
                "Quarantine affected batch/material",
                "Notify Quality Assurance department",
                "Document incident in batch manufacturing record"
            ]
        
        for i, action in enumerate(actions, 1):
            elements.append(Paragraph(f"{i}. {action}", self.styles['PharmaBody']))
        
        return elements

    def _create_investigation_requirements(self, deviation_data):
        """Create investigation requirements section"""
        elements = []
        
        elements.append(Paragraph("INVESTIGATION REQUIREMENTS", self.styles['PharmaHeading']))
        
        requirements = deviation_data.get('investigation_requirements', [])
        root_causes = deviation_data.get('potential_root_causes', [])
        
        if not requirements:
            requirements = [
                "Root Cause Analysis using 5 Whys methodology",
                "Equipment calibration verification",
                "Process parameter review",
                "Operator competency assessment"
            ]
        
        elements.append(Paragraph("<b>Required Investigations:</b>", self.styles['PharmaBody']))
        for req in requirements:
            elements.append(Paragraph(f"• {req}", self.styles['PharmaBody']))
        
        if root_causes:
            elements.append(Spacer(1, 0.05*inch))
            elements.append(Paragraph("<b>Potential Root Causes:</b>", self.styles['PharmaBody']))
            for cause in root_causes:
                elements.append(Paragraph(f"• {cause}", self.styles['PharmaBody']))
        
        return elements

    def _create_training_recommendations(self, deviation_data):
        """Create training recommendations section"""
        elements = []
        
        elements.append(Paragraph("TRAINING & COMPETENCY DEVELOPMENT", self.styles['PharmaHeading']))
        
        training = deviation_data.get('training_implications', {})
        training_recs = deviation_data.get('training_recommendations', {})
        
        if training.get('needs_retraining', False):
            elements.append(Paragraph("<b>Retraining Required:</b> YES", self.styles['PharmaBody']))
            elements.append(Paragraph(f"<b>Urgency:</b> {training.get('training_urgency', 'Unknown').title()}", self.styles['PharmaBody']))
            elements.append(Paragraph(f"<b>Affected Roles:</b> {', '.join(training.get('affected_roles', ['Operator', 'Supervisor']))}", self.styles['PharmaBody']))
        
        programs = training_recs.get('recommended_training_programs', [])
        if programs:
            elements.append(Spacer(1, 0.05*inch))
            elements.append(Paragraph("<b>Recommended Training Programs:</b>", self.styles['PharmaBody']))
            
            for program in programs[:2]:
                elements.append(Paragraph(f"<b>• {program.get('program_name', 'Unnamed Program')}</b>", self.styles['PharmaBody']))
                elements.append(Paragraph(f"  Target: {', '.join(program.get('target_audience', []))}", self.styles['PharmaBody']))
                elements.append(Paragraph(f"  Duration: {program.get('duration', 'Unknown')}", self.styles['PharmaBody']))
        
        return elements

    def _create_sop_references(self, contexts):
        """Create SOP references section"""
        elements = []
        
        elements.append(Paragraph("RELEVANT SOP REFERENCES", self.styles['PharmaHeading']))
        
        if contexts:
            for chunk, file in contexts:
                clean_filename = file.replace('_', ' ').replace('.pdf', '').title()
                elements.append(Paragraph(f"• {clean_filename}", self.styles['PharmaBody']))
        else:
            elements.append(Paragraph("No specific SOP references available", self.styles['PharmaBody']))
        
        return elements

    def _create_footer(self):
        """Create report footer"""
        footer_text = """
        <i>This is an automatically generated preliminary deviation report. 
        Formal investigation, documentation, and Quality Assurance approval are required.
        This report should be reviewed and supplemented with additional investigation findings.</i>
        
        <b>Confidential Pharmaceutical Document - For Internal Use Only</b>
        """
        
        return Paragraph(footer_text, self.styles['PharmaBody'])

    def _get_risk_style(self, severity):
        """Get risk level based on severity"""
        risk_map = {
            'critical': 'High',
            'major': 'High', 
            'minor': 'Medium',
            'observation': 'Low'
        }
        return risk_map.get(severity, 'Medium')

# Initialize PDF generator
pdf_generator = PDFReportGenerator()

# ========== CORE FUNCTIONS ==========
def embed_text(text):
    return embedder.encode(text, normalize_embeddings=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):
            break
    return chunks

def store_sop_chunk(file_name, chunk_index, chunk_text, embedding):
    try:
        key = f"sop:{file_name}:{chunk_index}"
        r.set(f"{key}:text", chunk_text.encode('utf-8'))
        r.set(f"{key}:file", file_name.encode('utf-8'))
        r.set(f"{key}:vector", embedding.astype(np.float32).tobytes())
        return True
    except Exception as e:
        print(f"❌ Failed to store SOP {key}: {e}")
        return False

def store_deviation_chunk(file_name, chunk_index, chunk_text, embedding):
    try:
        key = f"deviation:{file_name}:{chunk_index}"
        r.set(f"{key}:text", chunk_text.encode('utf-8'))
        r.set(f"{key}:file", file_name.encode('utf-8'))
        r.set(f"{key}:vector", embedding.astype(np.float32).tobytes())
        return True
    except Exception as e:
        print(f"❌ Failed to store deviation {key}: {e}")
        return False

def get_sop_chunk(key_base):
    try:
        text = r.get(f"{key_base}:text")
        file_name = r.get(f"{key_base}:file")
        vector = r.get(f"{key_base}:vector")
        
        if text and file_name and vector:
            return {
                "text": text.decode('utf-8'),
                "file": file_name.decode('utf-8'),
                "vector": np.frombuffer(vector, dtype=np.float32)
            }
        return None
    except Exception as e:
        print(f"❌ Failed to retrieve {key_base}: {e}")
        return None

def get_deviation_chunk(key_base):
    try:
        text = r.get(f"{key_base}:text")
        file_name = r.get(f"{key_base}:file")
        vector = r.get(f"{key_base}:vector")
        
        if text and file_name and vector:
            return {
                "text": text.decode('utf-8'),
                "file": file_name.decode('utf-8'),
                "vector": np.frombuffer(vector, dtype=np.float32)
            }
        return None
    except Exception as e:
        print(f"❌ Failed to retrieve {key_base}: {e}")
        return None

def ingest_sops():
    print("🧹 Clearing existing SOP Redis data...")
    # Clear only SOP data
    sop_keys = r.keys("sop:*")
    if sop_keys:
        r.delete(*sop_keys)
    
    sop_files = [f for f in os.listdir(SOP_FOLDER) if f.endswith(".pdf")]
    
    if not sop_files:
        print("❌ No PDF files found in SOP folder!")
        return
    
    print(f"📚 Found {len(sop_files)} SOP files to process...")
    
    for file in tqdm(sop_files, desc="Processing SOPs"):
        path = os.path.join(SOP_FOLDER, file)
        text = extract_text_from_pdf(path)
        
        if not text.strip():
            continue
            
        chunks = chunk_text(text)
        
        if not chunks:
            continue
            
        for i, chunk in enumerate(chunks):
            emb = embed_text(chunk).astype(np.float32)
            store_sop_chunk(file, i, chunk, emb)
                
    print("✅ SOP ingestion completed!")

def create_sample_deviation_reports():
    """Create sample deviation reports for analysis"""
    sample_reports = [
        {
            "title": "Critical Temperature Excursion",
            "content": """
            CRITICAL DEVIATION REPORT: TEMPERATURE EXCURSION IN API STORAGE
            
            Deviation ID: DEV-2024-001
            Severity: CRITICAL
            Category: Environmental/Storage
            
            Incident: Temperature excursion detected in raw material storage area RM-05. 
            The environmental monitoring system recorded temperatures of 12°C for 4 hours 
            against the required storage condition of 2-8°C for hygroscopic materials.
            
            Affected Materials: 
            - Batch #MAT-567 of Active Pharmaceutical Ingredient (Stability compromised)
            - Batch #EXC-890 of critical excipient
            
            Root Cause: HVAC system malfunction combined with operator failure to acknowledge alarm.
            Immediate Impact: Potential product quality impact requiring stability testing.
            
            CAPA: 
            - Immediate quarantine of affected materials
            - HVAC system maintenance and calibration
            - Operator retraining on alarm response procedures
            - Enhanced environmental monitoring frequency
            
            Regulatory Impact: Potential FDA 483 observation for inadequate controls.
            """
        },
        {
            "title": "Major Equipment Failure",
            "content": """
            MAJOR DEVIATION REPORT: COMPRESSION MACHINE FAILURE
            
            Deviation ID: DEV-2024-002
            Severity: MAJOR
            Category: Equipment/Manufacturing
            
            Incident: Compression machine CM-02 showed 8% deviation from calibrated pressure settings
            during routine performance qualification. This affected tablet hardness uniformity.
            
            Affected Batch: Batch #TAB-456 showed 15% out-of-specification tablets
            Batch Status: ON HOLD pending investigation
            
            Root Cause: Inadequate preventive maintenance schedule and calibration drift.
            Impact: Product quality impacted, potential batch rejection.
            
            CAPA:
            - Revised preventive maintenance schedule
            - Enhanced calibration frequency from monthly to weekly
            - Operator training on equipment monitoring
            - Implementation of real-time pressure monitoring
            
            Training Required: Equipment operation and monitoring for all operators.
            """
        },
        {
            "title": "Documentation Error Pattern",
            "content": """
            TREND ANALYSIS: DOCUMENTATION ERRORS
            
            Deviation ID: DEV-2024-003
            Severity: MINOR (but recurring pattern)
            Category: Documentation/Training
            
            Incident: Multiple documentation errors found in batch manufacturing records 
            over past 30 days. Missing signatures and incomplete entries in 5 different batches.
            
            Pattern: Recurring issue across multiple operators
            Root Cause: Inadequate training on Good Documentation Practices (GDP)
            
            Affected Departments:
            - Manufacturing operators
            - Quality control reviewers
            - Batch release team
            
            Trend: This is the 3rd similar deviation in 45 days indicating systematic training gap.
            
            CAPA:
            - Comprehensive GDP training for all personnel
            - Implementation of electronic batch records
            - Enhanced supervisory review process
            - Monthly documentation audits
            
            Regulatory Reference: FDA 21 CFR 211.100 and 211.192
            """
        },
        {
            "title": "Environmental Monitoring Failure",
            "content": """
            DEVIATION REPORT: ENVIRONMENTAL MONITORING FAILURE
            
            Deviation ID: DEV-2024-004
            Severity: MAJOR
            Category: Environmental/Quality Control
            
            Incident: Environmental monitoring in Grade C area showed particle count exceedance
            during aseptic filling operation. Count reached 352,000 particles vs limit of 350,000.
            
            Impact: Potential impact on product sterility assurance
            Batch Status: Quarantined for additional testing
            
            Root Cause: HVAC filter maintenance overdue and improper gowning procedure
            Immediate Actions: Stop manufacturing in affected area, enhanced cleaning
            
            CAPA:
            - HVAC filter replacement and validation
            - Gowning qualification for all operators
            - Increased environmental monitoring points
            - Revised cleaning validation protocol
            """
        }
    ]
    
    for i, report in enumerate(sample_reports):
        filename = f"sample_deviation_{i+1}.txt"
        filepath = os.path.join(DEVIATION_SAMPLE_FOLDER, filename)
        
        # Create text file as sample
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report['content'])
        
        print(f"✅ Created sample deviation report: {filename}")

def ingest_deviation_samples():
    print("📊 Processing deviation samples...")
    # Clear only deviation data
    deviation_keys = r.keys("deviation:*")
    if deviation_keys:
        r.delete(*deviation_keys)
    
    deviation_files = [f for f in os.listdir(DEVIATION_SAMPLE_FOLDER) if f.endswith(".txt")]
    
    if not deviation_files:
        print("❌ No deviation sample files found!")
        # Create sample deviation reports
        create_sample_deviation_reports()
        deviation_files = [f for f in os.listdir(DEVIATION_SAMPLE_FOLDER) if f.endswith(".txt")]
    
    print(f"📈 Found {len(deviation_files)} deviation sample files to process...")
    
    for file in tqdm(deviation_files, desc="Processing Deviation Samples"):
        path = os.path.join(DEVIATION_SAMPLE_FOLDER, file)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text.strip():
            continue
            
        chunks = chunk_text(text)
        
        if not chunks:
            continue
            
        for i, chunk in enumerate(chunks):
            emb = embed_text(chunk).astype(np.float32)
            store_deviation_chunk(file, i, chunk, emb)
                
    print("✅ Deviation samples ingestion completed!")

def search_sops(query, top_k=TOP_K):
    """Search SOP documents only"""
    try:
        query_emb = embed_text([query])[0]
        all_keys = r.keys("sop:*:text")
        
        if not all_keys:
            return []
        
        results = []
        
        for key in all_keys:
            base_key = key.decode('utf-8').rsplit(":", 1)[0]
            data = get_sop_chunk(base_key)
            
            if data:
                vec = data["vector"]
                chunk_text = data["text"]
                file_name = data["file"]
                
                score = np.dot(query_emb, vec) / (np.linalg.norm(query_emb) * np.linalg.norm(vec))
                results.append((score, chunk_text, file_name))
        
        results.sort(reverse=True, key=lambda x: x[0])
        return [(chunk, file) for score, chunk, file in results[:top_k] if score > 0.3]
        
    except Exception as e:
        print(f"❌ SOP search failed: {e}")
        return []

def search_deviations(query, top_k=TOP_K):
    """Search deviation reports only"""
    try:
        query_emb = embed_text([query])[0]
        all_keys = r.keys("deviation:*:text")
        
        if not all_keys:
            return []
        
        results = []
        
        for key in all_keys:
            base_key = key.decode('utf-8').rsplit(":", 1)[0]
            data = get_deviation_chunk(base_key)
            
            if data:
                vec = data["vector"]
                chunk_text = data["text"]
                file_name = data["file"]
                
                score = np.dot(query_emb, vec) / (np.linalg.norm(query_emb) * np.linalg.norm(vec))
                results.append((score, chunk_text, file_name))
        
        results.sort(reverse=True, key=lambda x: x[0])
        return [(chunk, file) for score, chunk, file in results[:top_k] if score > 0.3]
        
    except Exception as e:
        print(f"❌ Deviation search failed: {e}")
        return []

def build_prompt(query, contexts):
    """Build prompt for SOP-based answers - conversational without references"""
    if contexts:
        context_text = "\n\n".join([chunk for chunk, file in contexts])
        
        # Extract SOP name for conversational reference
        sop_names = list(set([file.replace('_', ' ').replace('.pdf', '') for chunk, file in contexts]))
        sop_reference = f" according to {sop_names[0]}" if sop_names else ""
        
        return f"""You are a pharmaceutical compliance expert. Answer STRICTLY based on the provided SOP content only.

SOP CONTENT:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
1. Answer ONLY using the provided SOP content
2. Provide a direct, conversational answer without mentioning "SOP" or "document" repeatedly
3. If the SOP content doesn't contain the answer, say "This information is not available in the current procedures"
4. Be precise and technical but conversational
5. Do not list references or file names in the answer

ANSWER{sop_reference}:"""
    else:
        return f"""QUESTION: {query}

ANSWER: This information is not available in the current procedures. Please ensure relevant SOPs are uploaded and processed."""

def call_groq(prompt):
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
        
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API Error {resp.status_code}: {resp.text}")
    except Exception as e:
        return f"Error: {e}"

def detect_deviation(query, contexts):
    """Detect deviations using pharmaceutical standards"""
    try:
        deviation_prompt = f"""
        Analyze this pharmaceutical incident for compliance deviations:

        INCIDENT: {query}
        RELEVANT SOP CONTEXT: {contexts}

        Analyze and return JSON with deviation analysis:

        {{
            "is_deviation": boolean,
            "deviation_type": "planned/unplanned",
            "severity_level": "critical/major/minor",
            "deviation_category": "equipment/process/documentation/training/environmental/material",
            "stage_of_occurrence": "manufacturing/packaging/storage/testing/other",
            "risk_assessment": {{
                "product_quality_impact": "confirmed/potential/none",
                "patient_safety_impact": "high/medium/low/none", 
                "regulatory_impact": "high/medium/low",
                "business_impact": "high/medium/low"
            }},
            "immediate_actions": [
                "list of immediate containment actions"
            ],
            "investigation_requirements": [
                "required investigation steps"
            ],
            "root_cause_categories": ["human_error", "equipment_failure", "procedural_gap", "training_deficiency", "environmental", "material_issue"],
            "training_implications": {{
                "needs_retraining": boolean,
                "affected_roles": ["list of roles"],
                "training_urgency": "immediate/within_week/within_month"
            }},
            "regulatory_references": ["FDA 21 CFR 211.100", "FDA 21 CFR 211.192", "EU GMP Chapter 1", "EU GMP Chapter 8"],
            "confidence_score": float
        }}
        """

        response = call_groq(deviation_prompt)
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            return json.loads(response[start_idx:end_idx])
    except Exception as e:
        print(f"⚠️ Deviation detection failed: {e}")
    
    # Enhanced fallback response
    return {
        "is_deviation": True,
        "deviation_type": "unplanned",
        "severity_level": "major",
        "deviation_category": "environmental",
        "stage_of_occurrence": "storage",
        "risk_assessment": {
            "product_quality_impact": "potential",
            "patient_safety_impact": "medium", 
            "regulatory_impact": "high",
            "business_impact": "medium"
        },
        "immediate_actions": [
            "Quarantine affected materials",
            "Notify Quality Assurance immediately",
            "Document the deviation in batch records",
            "Assess impact on material stability"
        ],
        "investigation_requirements": [
            "Root cause analysis using 5 Whys",
            "Review environmental monitoring system logs",
            "Interview involved personnel",
            "Assess material stability data"
        ],
        "root_cause_categories": ["human_error", "equipment_failure", "procedural_gap"],
        "training_implications": {
            "needs_retraining": True,
            "affected_roles": ["warehouse_operators", "quality_control"],
            "training_urgency": "within_week"
        },
        "regulatory_references": ["FDA 21 CFR 211.100", "FDA 21 CFR 211.192", "EU GMP Chapter 1"],
        "confidence_score": 0.85
    }

def get_real_time_alerts():
    """Get real-time compliance alerts from critical deviations"""
    try:
        # Check for recent critical deviations
        critical_deviations = flag_critical_deviations()
        
        alerts = []
        for i, deviation in enumerate(critical_deviations[:5]):  # Top 5 most critical
            alerts.append({
                "alert_id": f"ALERT-{datetime.now().strftime('%H%M%S')}-{i}",
                "type": "critical_deviation",
                "title": "Critical Deviation Flagged",
                "description": deviation['content'],
                "severity": deviation['analysis']['risk_level'],
                "immediate_actions": deviation['analysis']['recommended_actions'],
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
        
    except Exception as e:
        print(f"❌ Alert generation failed: {e}")
        return []

def flag_critical_deviations():
    """Flag critical and major deviations for immediate attention"""
    try:
        # Search for high-risk deviations
        high_risk_queries = [
            "critical deviation major product quality impact patient safety",
            "temperature excursion stability impact",
            "equipment failure batch rejection",
            "regulatory compliance failure",
            "out of specification OOS result",
            "batch rejection quality failure"
        ]
        
        flagged_deviations = []
        
        for query in high_risk_queries:
            contexts = search_deviations(query, top_k=2)
            for chunk, file in contexts:
                # Analyze if this indicates a critical pattern
                analysis_prompt = f"""
                Analyze this deviation content for critical risk factors:
                
                CONTENT: {chunk}
                
                Return JSON analysis:
                {{
                    "is_critical": boolean,
                    "risk_level": "critical/major/moderate",
                    "immediate_attention_required": boolean,
                    "affected_areas": ["list of departments/systems"],
                    "potential_impact": "description",
                    "recommended_actions": ["list of actions"]
                }}
                """
                
                try:
                    analysis = call_groq(analysis_prompt)
                    start_idx = analysis.find('{')
                    end_idx = analysis.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        analysis_data = json.loads(analysis[start_idx:end_idx])
                        if analysis_data.get("is_critical", False) or analysis_data.get("risk_level") in ["critical", "major"]:
                            flagged_deviations.append({
                                "content": chunk[:200] + "...",
                                "source_file": file,
                                "analysis": analysis_data,
                                "timestamp": datetime.now().isoformat()
                            })
                except Exception as e:
                    print(f"⚠️ Analysis failed for chunk: {e}")
                    continue
        
        return flagged_deviations
        
    except Exception as e:
        print(f"❌ Critical deviation flagging failed: {e}")
        return []

def analyze_non_compliance_trends():
    """Analyze patterns of non-compliance across deviations"""
    try:
        # Search for common non-compliance patterns
        trend_patterns = [
            "recurring deviation same root cause",
            "training deficiency multiple incidents", 
            "equipment failure repeated",
            "documentation error frequent",
            "environmental monitoring failure pattern",
            "human error procedure not followed"
        ]
        
        trends = []
        
        for pattern in trend_patterns:
            contexts = search_deviations(pattern, top_k=3)
            if contexts:
                trend_analysis_prompt = f"""
                Analyze these deviation patterns for systematic non-compliance:
                
                PATTERN: {pattern}
                DEVIATION CONTEXTS: {contexts}
                
                Return JSON trend analysis:
                {{
                    "trend_identified": boolean,
                    "trend_type": "training/equipment/documentation/process/environmental/human_error",
                    "severity": "high/medium/low",
                    "recurrence_frequency": "description",
                    "root_cause_pattern": "description",
                    "departments_affected": ["list"],
                    "risk_implications": "description",
                    "preventive_measures": ["list of measures"]
                }}
                """
                
                try:
                    analysis = call_groq(trend_analysis_prompt)
                    start_idx = analysis.find('{')
                    end_idx = analysis.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        trend_data = json.loads(analysis[start_idx:end_idx])
                        if trend_data.get("trend_identified", False):
                            trends.append({
                                "pattern": pattern,
                                "analysis": trend_data,
                                "supporting_evidence": [chunk[:150] + "..." for chunk, file in contexts],
                                "timestamp": datetime.now().isoformat()
                            })
                except Exception as e:
                    print(f"⚠️ Trend analysis failed: {e}")
                    continue
        
        return trends
        
    except Exception as e:
        print(f"❌ Non-compliance trend analysis failed: {e}")
        return []

def generate_compliance_dashboard():
    """Generate comprehensive compliance dashboard data"""
    try:
        # Get flagged critical deviations
        critical_deviations = flag_critical_deviations()
        
        # Get non-compliance trends
        compliance_trends = analyze_non_compliance_trends()
        
        # Calculate compliance metrics
        total_deviations = len(glob.glob(os.path.join(DEVIATION_FOLDER, "*.txt"))) + \
                          len(glob.glob(os.path.join(DEVIATION_SAMPLE_FOLDER, "*.txt")))
        
        # Calculate severity distribution
        critical_count = len([d for d in critical_deviations if d['analysis']['risk_level'] == 'critical'])
        major_count = len([d for d in critical_deviations if d['analysis']['risk_level'] == 'major'])
        minor_count = max(0, total_deviations - critical_count - major_count)
        
        critical_percentage = (critical_count / total_deviations * 100) if total_deviations > 0 else 0
        compliance_score = max(0, 100 - (critical_count * 10 + major_count * 5 + minor_count * 2))
        
        severity_data = {
            "critical_count": critical_count,
            "major_count": major_count, 
            "minor_count": minor_count,
            "critical_percentage": round(critical_percentage, 1),
            "compliance_score": min(100, max(0, round(compliance_score, 1)))
        }
        
        dashboard_data = {
            "dashboard_id": f"DASH-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_deviations_analyzed": total_deviations,
                "critical_deviations_flagged": len(critical_deviations),
                "non_compliance_trends_identified": len(compliance_trends),
                "severity_distribution": severity_data,
                "overall_compliance_score": severity_data.get("compliance_score", 85)
            },
            "critical_deviations": critical_deviations,
            "compliance_trends": compliance_trends,
            "recommendations": generate_dashboard_recommendations(critical_deviations, compliance_trends)
        }
        
        return dashboard_data
        
    except Exception as e:
        print(f"❌ Dashboard generation failed: {e}")
        return {"error": str(e)}

def generate_dashboard_recommendations(critical_deviations, compliance_trends):
    """Generate actionable recommendations from analysis"""
    try:
        # Extract key issues for recommendations
        critical_issues = [dev['analysis']['risk_level'] for dev in critical_deviations]
        trend_types = [trend['analysis']['trend_type'] for trend in compliance_trends]
        
        recommendation_prompt = f"""
        Based on this pharmaceutical compliance analysis:
        
        CRITICAL DEVIATIONS: {len(critical_deviations)} flagged issues with risk levels: {critical_issues}
        COMPLIANCE TRENDS: {len(compliance_trends)} identified trends: {trend_types}
        
        Generate actionable pharmaceutical GMP recommendations in JSON format:
        {{
            "immediate_actions": ["list of 3-5 urgent actions for quality team"],
            "preventive_measures": ["list of 3-5 preventive measures"],
            "training_priorities": ["list of 3-5 training needs with departments"],
            "system_improvements": ["list of 3-5 system enhancements"],
            "monitoring_enhancements": ["list of 3-5 monitoring improvements"]
        }}
        
        Focus on FDA 21 CFR Part 211 and EU GMP compliance.
        """
        
        recommendations = call_groq(recommendation_prompt)
        start_idx = recommendations.find('{')
        end_idx = recommendations.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            return json.loads(recommendations[start_idx:end_idx])
        else:
            # Fallback recommendations
            return {
                "immediate_actions": [
                    "Review all critical deviations with Quality Assurance",
                    "Quarantine affected batches mentioned in deviations",
                    "Conduct immediate equipment calibration checks"
                ],
                "preventive_measures": [
                    "Strengthen training programs on GDP and GMP",
                    "Implement automated environmental monitoring",
                    "Enhance documentation review processes"
                ],
                "training_priorities": [
                    "Good Documentation Practices for all operators",
                    "Equipment operation and maintenance training",
                    "Deviation reporting and investigation procedures"
                ],
                "system_improvements": [
                    "Upgrade to electronic batch records system",
                    "Implement real-time monitoring alerts",
                    "Enhance change control procedures"
                ],
                "monitoring_enhancements": [
                    "Increase environmental monitoring frequency",
                    "Implement trend analysis dashboard",
                    "Enhance audit trail review processes"
                ]
            }
            
    except Exception as e:
        print(f"❌ Recommendation generation failed: {e}")
        return {
            "immediate_actions": ["Review critical deviations immediately"],
            "preventive_measures": ["Implement enhanced monitoring"],
            "training_priorities": ["Conduct GMP refresher training"],
            "system_improvements": ["Review and update procedures"],
            "monitoring_enhancements": ["Increase audit frequency"]
        }

def analyze_deviation_trends(days: int = 30):
    """Analyze deviation trends using both current and historical data"""
    try:
        # Search for similar historical deviations
        recent_deviations = []
        
        # Get current deviation reports
        deviation_files = glob.glob(os.path.join(DEVIATION_FOLDER, "*.txt"))
        for file in deviation_files[:5]:  # Recent 5 reports
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    recent_deviations.append(content[:1000])
            except:
                continue
        
        # Search historical deviation samples for patterns
        trend_query = "common root causes training deficiencies equipment failures"
        historical_contexts = search_deviations(trend_query, top_k=5)
        
        trends_prompt = f"""
        Analyze pharmaceutical deviation trends based on:
        
        RECENT DEVIATIONS ({len(recent_deviations)} reports):
        {recent_deviations}
        
        HISTORICAL PATTERNS:
        {historical_contexts}
        
        Provide comprehensive trend analysis covering:
        1. Most common deviation categories
        2. Recurring root causes
        3. Training gaps identified
        4. Equipment/systemic issues
        5. Recommended preventive actions
        
        Focus on actionable insights for quality improvement.
        """
        
        trends_analysis = call_groq(trends_prompt)
        
        # Generate specific retraining recommendations
        training_prompt = f"""
        Based on these deviation trends, generate specific retraining recommendations:
        
        {trends_analysis}
        
        Provide structured training program suggestions including:
        - Target audiences
        - Training topics
        - Urgency levels
        - Expected outcomes
        """
        
        training_recommendations = call_groq(training_prompt)
        
        analysis_id = f"TRENDS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        return {
            "analysis_id": analysis_id,
            "trends_analysis": trends_analysis,
            "training_recommendations": training_recommendations,
            "reports_analyzed": len(recent_deviations),
            "historical_patterns_used": len(historical_contexts)
        }
        
    except Exception as e:
        return {"error": f"Trends analysis failed: {str(e)}"}

def generate_retraining_suggestions(deviation_data=None):
    """Generate retraining suggestions based on deviation analysis"""
    try:
        # Search for relevant training-related deviations
        training_contexts = search_deviations("training retraining competency", top_k=3)
        
        prompt = """
        Based on pharmaceutical GMP compliance requirements and common training-related deviations,
        generate comprehensive retraining program suggestions covering:
        
        - Equipment operation and maintenance
        - Documentation practices and GDP
        - Quality control procedures
        - Regulatory compliance awareness
        - Good Manufacturing Practices
        - Specific technical skills based on deviation patterns
        
        Provide detailed training program outlines with:
        - Program objectives
        - Target audiences by department
        - Duration and delivery methods
        - Assessment criteria
        - Expected competency outcomes
        """
        
        if training_contexts:
            prompt += f"\n\nRELEVANT DEVIATION PATTERNS:\n{training_contexts}"
        
        suggestions = call_groq(prompt)
        
        program_id = f"TRAIN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Save to file
        text_filename = f"{program_id}_SUGGESTIONS.txt"
        text_path = os.path.join(TRAINING_FOLDER, text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(suggestions)
        
        return {
            "program_id": program_id,
            "suggestions": suggestions,
            "text_report_path": text_path
        }
        
    except Exception as e:
        return {"error": f"Retraining suggestions failed: {str(e)}"}

def generate_deviation_report(deviation_data, query, contexts):
    """Generate comprehensive deviation report with PDF"""
    deviation_id = f"DEV-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Generate detailed text report
    report_prompt = f"""
    Generate a comprehensive pharmaceutical deviation report following GMP compliance standards:

    DEVIATION ID: {deviation_id}
    INCIDENT: {query}
    DETECTION DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ANALYSIS DATA:
    {json.dumps(deviation_data, indent=2)}

    Create detailed report with:
    1. Executive Summary
    2. Deviation Classification
    3. Detailed Event Description  
    4. Immediate Actions Taken
    5. Impact Assessment
    6. Root Cause Analysis
    7. Corrective and Preventive Actions (CAPA)
    8. Training Implications
    9. Regulatory Compliance
    10. Closure Requirements

    Focus on pharmaceutical GMP compliance and regulatory requirements.
    """
    
    try:
        text_report = call_groq(report_prompt)
        
        # Save text report
        text_filename = f"{deviation_id}_DETAILED.txt"
        text_path = os.path.join(DEVIATION_FOLDER, text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        # Generate PDF report
        pdf_path = pdf_generator.create_deviation_report_pdf(deviation_data, query, contexts, deviation_id)
        
        return {
            "deviation_id": deviation_id,
            "text_report_path": text_path,
            "pdf_report_path": pdf_path,
            "summary": text_report[:500] + "..." if len(text_report) > 500 else text_report
        }
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return None

# ========== FASTAPI BACKEND ==========
app = FastAPI(title="Pharma SOP AI Compliance Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
def process_sops():
    """Process and ingest all SOP documents"""
    try:
        ingest_sops()
        return {"status": "success", "message": "SOPs processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process-deviations")
def process_deviation_samples():
    """Process and ingest deviation samples"""
    try:
        ingest_deviation_samples()
        return {"status": "success", "message": "Deviation samples processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deviation processing failed: {str(e)}")

@app.post("/upload-sop")
async def upload_sop(file: UploadFile = File(...)):
    """Upload and store SOP PDF"""
    try:
        # Save uploaded file
        file_path = os.path.join(SOP_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"status": "success", "message": f"SOP {file.filename} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/list-sops")
def list_sops():
    """List all available SOP files"""
    try:
        sop_files = [f for f in os.listdir(SOP_FOLDER) if f.endswith(".pdf")]
        return {"sops": sop_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list SOPs: {str(e)}")

@app.post("/query")
def query_sop(query: str = Form(...)):
    """Query SOP system - conversational answers without references"""
    try:
        contexts = search_sops(query)
        prompt = build_prompt(query, contexts)
        answer = call_groq(prompt)
        
        # Extract SOP names for frontend display only (not in answer)
        sop_references = list(set([file.replace('_', ' ').replace('.pdf', '') for _, file in contexts]))
        
        return {
            "answer": answer,
            "contexts_used": len(contexts),
            "sop_references": sop_references  # For frontend display only
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/report-incident")
def report_incident(incident: str = Form(...), generate_pdf: bool = Form(True)):
    """Report and analyze a compliance incident"""
    try:
        contexts = search_sops(incident)
        deviation_data = detect_deviation(incident, contexts)
        
        response = {
            "incident": incident,
            "is_deviation": deviation_data.get('is_deviation', False),
            "deviation_analysis": deviation_data,
            "sop_references": [file.replace('_', ' ').replace('.pdf', '') for _, file in contexts]
        }
        
        # Generate report if deviation detected and PDF requested
        if deviation_data.get('is_deviation', False) and generate_pdf:
            report = generate_deviation_report(deviation_data, incident, contexts)
            if report:
                response["report"] = report
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Incident reporting failed: {str(e)}")

@app.post("/deviation-prompt")
def create_deviation_from_prompt(
    incident_description: str = Form(...),
    severity: str = Form("major"),
    category: str = Form("process"),
    generate_pdf: bool = Form(True)
):
    """Create deviation report from structured prompt"""
    try:
        contexts = search_sops(incident_description)
        
        # Create structured deviation data
        deviation_data = {
            "is_deviation": True,
            "deviation_type": "unplanned",
            "severity_level": severity,
            "deviation_category": category,
            "stage_of_occurrence": "manufacturing",
            "risk_assessment": {
                "product_quality_impact": "high" if severity in ["critical", "major"] else "medium",
                "patient_safety_impact": "medium" if severity == "critical" else "low",
                "regulatory_impact": "high" if severity in ["critical", "major"] else "medium",
                "business_impact": "medium"
            },
            "immediate_actions": [
                "Investigate root cause",
                "Document incident",
                "Notify relevant departments",
                "Quarantine affected materials if applicable"
            ],
            "investigation_requirements": [
                "Root cause analysis using 5 Whys methodology",
                "Review relevant documentation",
                "Interview involved personnel"
            ],
            "training_implications": {
                "needs_retraining": True,
                "affected_roles": ["operators", "supervisors", "quality_personnel"],
                "training_urgency": "immediate" if severity == "critical" else "within_week"
            }
        }
        
        response = {
            "incident": incident_description,
            "deviation_analysis": deviation_data,
            "sop_references": [file.replace('_', ' ').replace('.pdf', '') for _, file in contexts]
        }
        
        if generate_pdf:
            report = generate_deviation_report(deviation_data, incident_description, contexts)
            if report:
                response["report"] = report
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deviation creation failed: {str(e)}")

@app.get("/real-time-alerts")
def get_real_time_alerts_endpoint():
    """Get real-time compliance alerts"""
    try:
        alerts = get_real_time_alerts()
        return {
            "status": "success",
            "alerts_count": len(alerts),
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert generation failed: {str(e)}")

@app.get("/flag-critical-deviations")
def flag_critical_deviations_endpoint():
    """Flag critical deviations for immediate attention"""
    try:
        flagged_deviations = flag_critical_deviations()
        return {
            "status": "success",
            "flagged_count": len(flagged_deviations),
            "critical_deviations": flagged_deviations,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Critical deviation flagging failed: {str(e)}")

@app.get("/compliance-trends")
def get_compliance_trends_endpoint():
    """Analyze non-compliance trends"""
    try:
        trends = analyze_non_compliance_trends()
        return {
            "status": "success", 
            "trends_identified": len(trends),
            "compliance_trends": trends,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compliance trend analysis failed: {str(e)}")

@app.get("/compliance-dashboard")
def get_compliance_dashboard_endpoint():
    """Generate comprehensive compliance dashboard"""
    try:
        dashboard = generate_compliance_dashboard()
        return {
            "status": "success",
            "dashboard": dashboard
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")

@app.get("/deviation-trends")
def get_deviation_trends(days: int = 30):
    """Analyze deviation trends and patterns"""
    try:
        trends = analyze_deviation_trends(days)
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trends analysis failed: {str(e)}")

@app.get("/retraining-suggestions")
def get_retraining_suggestions(deviation_id: str = None):
    """Generate retraining suggestions based on deviations"""
    try:
        suggestions = generate_retraining_suggestions()
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining suggestions failed: {str(e)}")

@app.get("/download-pdf/{pdf_filename}")
def download_pdf(pdf_filename: str):
    """Download generated PDF report"""
    pdf_path = os.path.join(PDF_REPORTS_FOLDER, pdf_filename)
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type='application/pdf', filename=pdf_filename)
    else:
        raise HTTPException(status_code=404, detail="PDF file not found")

@app.get("/list-reports")
def list_reports():
    """List all generated reports"""
    try:
        reports = {
            "deviation_reports": [],
            "training_reports": [],
            "trends_reports": []
        }
        
        # Get PDF reports from PDF_REPORTS_FOLDER
        pdf_files = glob.glob(os.path.join(PDF_REPORTS_FOLDER, "*.pdf"))
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            if filename.startswith("DEV"):
                reports["deviation_reports"].append(filename)
            elif filename.startswith("TRAIN"):
                reports["training_reports"].append(filename)
            elif filename.startswith("TRENDS"):
                reports["trends_reports"].append(filename)
        
        # Also include text reports from DEVIATION_FOLDER for completeness
        text_files = glob.glob(os.path.join(DEVIATION_FOLDER, "*.txt"))
        for text_file in text_files:
            filename = os.path.basename(text_file)
            if filename.startswith("DEV") and filename not in [r.replace('.pdf', '.txt') for r in reports["deviation_reports"]]:
                reports["deviation_reports"].append(filename)
        
        return reports
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        r.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    sop_files = len([f for f in os.listdir(SOP_FOLDER) if f.endswith(".pdf")]) if os.path.exists(SOP_FOLDER) else 0
    deviation_samples = len([f for f in os.listdir(DEVIATION_SAMPLE_FOLDER) if f.endswith(".txt")]) if os.path.exists(DEVIATION_SAMPLE_FOLDER) else 0
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "sop_files_available": sop_files,
        "deviation_samples_available": deviation_samples,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting Pharma SOP AI Compliance Backend...")
    print("📁 SOP Folder:", os.path.abspath(SOP_FOLDER))
    print("📊 Deviation Samples Folder:", os.path.abspath(DEVIATION_SAMPLE_FOLDER))
    print("🔑 Groq Model:", GROQ_MODEL)
    print("🗄️  Redis:", f"{REDIS_HOST}:{REDIS_PORT}")
    
    # Ensure all folders exist
    for folder in [SOP_FOLDER, DEVIATION_FOLDER, DEVIATION_SAMPLE_FOLDER, TRAINING_FOLDER, PDF_REPORTS_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    
    # Create and process deviation samples on startup
    print("📝 Setting up deviation samples...")
    create_sample_deviation_reports()
    ingest_deviation_samples()
    
    print("✅ Backend startup completed!")
    uvicorn.run(app, host="0.0.0.0", port=8000)