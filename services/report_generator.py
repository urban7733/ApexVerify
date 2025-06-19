from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from pathlib import Path
import json
from datetime import datetime
import logging

class ReportGenerator:
    def __init__(self):
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.styles = getSampleStyleSheet()
        
        # Add custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12
        ))
    
    def generate_pdf(self, file_id: str, analysis_data: dict) -> str:
        """
        Generate a PDF report for the analysis results
        
        Args:
            file_id: Unique identifier for the file
            analysis_data: Analysis results dictionary
            
        Returns:
            str: Path to the generated PDF file
        """
        try:
            # Create PDF file path
            pdf_path = self.reports_dir / f"report_{file_id}.pdf"
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Build PDF content
            story = []
            
            # Title
            story.append(Paragraph("Apex Verify AI - Deepfake Analysis Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Analysis Results
            story.append(Paragraph("Analysis Results", self.styles['CustomHeading']))
            
            # Create results table
            results_data = [
                ["Property", "Value"],
                ["Is Deepfake", str(analysis_data["is_deepfake"])],
                ["Confidence Score", f"{analysis_data['confidence_score']:.2%}"],
                ["Prediction", analysis_data["prediction"]],
                ["Analysis Timestamp", analysis_data["analysis_timestamp"]]
            ]
            
            results_table = Table(results_data, colWidths=[200, 300])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(results_table)
            story.append(Spacer(1, 20))
            
            # Model Information
            story.append(Paragraph("Model Information", self.styles['CustomHeading']))
            model_data = [
                ["Property", "Value"],
                ["Model Name", analysis_data["model_metadata"]["model_name"]],
                ["Model Type", analysis_data["model_metadata"]["model_type"]],
                ["Confidence Threshold", str(analysis_data["model_metadata"]["confidence_threshold"])]
            ]
            
            model_table = Table(model_data, colWidths=[200, 300])
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(model_table)
            
            # Build PDF
            doc.build(story)
            
            return str(pdf_path)
            
        except Exception as e:
            logging.error(f"Error generating PDF report: {str(e)}")
            raise 