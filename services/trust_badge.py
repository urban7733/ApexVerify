from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime
from typing import Dict, Optional
import logging

class TrustBadgeService:
    def __init__(self):
        self.badge_dir = "static/badges"
        os.makedirs(self.badge_dir, exist_ok=True)
        self.default_font = "Arial.ttf"  # You can replace this with your preferred font
        
    def generate_badge(self, 
                      analysis_result: Dict,
                      output_path: str,
                      badge_style: str = "standard") -> str:
        """
        Generate a trust badge for the analyzed media
        
        Args:
            analysis_result: Results from deepfake analysis
            output_path: Path to save the badge
            badge_style: Style of the badge (standard, minimal, detailed)
            
        Returns:
            Path to the generated badge
        """
        try:
            # Create badge image
            badge = Image.new('RGBA', (400, 200), (255, 255, 255, 0))
            draw = ImageDraw.Draw(badge)
            
            # Load font
            try:
                font = ImageFont.truetype(self.default_font, 24)
                small_font = ImageFont.truetype(self.default_font, 16)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw badge content based on style
            if badge_style == "minimal":
                self._draw_minimal_badge(draw, analysis_result, font)
            elif badge_style == "detailed":
                self._draw_detailed_badge(draw, analysis_result, font, small_font)
            else:  # standard
                self._draw_standard_badge(draw, analysis_result, font, small_font)
            
            # Save badge
            badge_path = os.path.join(self.badge_dir, f"badge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            badge.save(badge_path)
            
            return badge_path
            
        except Exception as e:
            logging.error(f"Error generating trust badge: {str(e)}")
            return None
    
    def _draw_standard_badge(self, draw: ImageDraw, analysis: Dict, font: ImageFont, small_font: ImageFont):
        """Draw standard trust badge"""
        # Draw border
        draw.rectangle([(0, 0), (399, 199)], outline=(0, 0, 0), width=2)
        
        # Draw title
        draw.text((20, 20), "Apex Verify AI", fill=(0, 0, 0), font=font)
        
        # Draw verification status
        status = "VERIFIED" if not analysis.get("is_deepfake", True) else "UNVERIFIED"
        status_color = (0, 128, 0) if status == "VERIFIED" else (255, 0, 0)
        draw.text((20, 60), status, fill=status_color, font=font)
        
        # Draw confidence score
        confidence = analysis.get("real_percentage", 0)
        draw.text((20, 100), f"Confidence: {confidence:.1f}%", fill=(0, 0, 0), font=small_font)
        
        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((20, 160), timestamp, fill=(0, 0, 0), font=small_font)
    
    def _draw_minimal_badge(self, draw: ImageDraw, analysis: Dict, font: ImageFont):
        """Draw minimal trust badge"""
        # Draw border
        draw.rectangle([(0, 0), (199, 99)], outline=(0, 0, 0), width=2)
        
        # Draw status
        status = "✓" if not analysis.get("is_deepfake", True) else "✗"
        status_color = (0, 128, 0) if status == "✓" else (255, 0, 0)
        draw.text((80, 40), status, fill=status_color, font=font)
    
    def _draw_detailed_badge(self, draw: ImageDraw, analysis: Dict, font: ImageFont, small_font: ImageFont):
        """Draw detailed trust badge"""
        # Draw border
        draw.rectangle([(0, 0), (599, 299)], outline=(0, 0, 0), width=2)
        
        # Draw title
        draw.text((20, 20), "Apex Verify AI - Detailed Analysis", fill=(0, 0, 0), font=font)
        
        # Draw verification status
        status = "VERIFIED" if not analysis.get("is_deepfake", True) else "UNVERIFIED"
        status_color = (0, 128, 0) if status == "VERIFIED" else (255, 0, 0)
        draw.text((20, 60), status, fill=status_color, font=font)
        
        # Draw confidence scores
        real_percent = analysis.get("real_percentage", 0)
        fake_percent = analysis.get("fake_percentage", 0)
        draw.text((20, 100), f"Real: {real_percent:.1f}%", fill=(0, 0, 0), font=small_font)
        draw.text((20, 130), f"Fake: {fake_percent:.1f}%", fill=(0, 0, 0), font=small_font)
        
        # Draw AI model information
        ai_model = analysis.get("ai_model_used", {})
        if ai_model.get("detected"):
            draw.text((20, 170), f"AI Model: {ai_model['model_name']}", fill=(0, 0, 0), font=small_font)
            draw.text((20, 200), f"Confidence: {ai_model['confidence']*100:.1f}%", fill=(0, 0, 0), font=small_font)
        
        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((20, 260), timestamp, fill=(0, 0, 0), font=small_font)
    
    def apply_badge_to_image(self, 
                           image_path: str,
                           badge_path: str,
                           output_path: str,
                           position: str = "bottom-right") -> str:
        """
        Apply trust badge to an image
        
        Args:
            image_path: Path to the original image
            badge_path: Path to the trust badge
            output_path: Path to save the watermarked image
            position: Position of the badge (top-left, top-right, bottom-left, bottom-right)
            
        Returns:
            Path to the watermarked image
        """
        try:
            # Open images
            image = Image.open(image_path)
            badge = Image.open(badge_path)
            
            # Convert to RGBA if needed
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            if badge.mode != 'RGBA':
                badge = badge.convert('RGBA')
            
            # Calculate badge position
            if position == "top-left":
                pos = (10, 10)
            elif position == "top-right":
                pos = (image.width - badge.width - 10, 10)
            elif position == "bottom-left":
                pos = (10, image.height - badge.height - 10)
            else:  # bottom-right
                pos = (image.width - badge.width - 10, image.height - badge.height - 10)
            
            # Create new image with badge
            watermarked = Image.new('RGBA', image.size, (0, 0, 0, 0))
            watermarked.paste(image, (0, 0))
            watermarked.paste(badge, pos, badge)
            
            # Save watermarked image
            watermarked.save(output_path)
            
            return output_path
            
        except Exception as e:
            logging.error(f"Error applying trust badge: {str(e)}")
            return None 