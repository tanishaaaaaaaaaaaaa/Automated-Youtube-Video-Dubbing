#!/usr/bin/env python3
"""
Fixed YouTube Speech Translator
Solves the file path issue in TTS generation
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import subprocess

# Core libraries
try:
    import yt_dlp
    import whisper
    from deep_translator import GoogleTranslator
    from gtts import gTTS
    from pydub import AudioSegment
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: pip install -r requirements.txt")
    sys.exit(1)

class FixedYouTubeTranslator:
    def __init__(self, output_dir: str = "output", temp_dir: str = "temp"):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        
        # Create directories with absolute paths
        self.output_dir = self.output_dir.resolve()
        self.temp_dir = self.temp_dir.resolve()
        
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Load Whisper model
        self.logger.info("🤖 Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model("base")
            self.logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to load Whisper model: {e}")
            sys.exit(1)
    
    def download_video_robust(self, url: str, video_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Robust video download with multiple strategies"""
        self.logger.info(f"📥 Downloading video: {url}")
        
        # Clean temp directory first
        self._clean_temp_files(video_id)
        
        video_filename = f"{video_id}.%(ext)s"
        video_path = self.temp_dir / video_filename
        
        # Multiple download strategies
        strategies = [
            # Strategy 1: Best quality MP4
            {
                'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
                'outtmpl': str(video_path),
                'quiet': True,
                'no_warnings': True
            },
            # Strategy 2: Lower quality, more compatible
            {
                'format': 'worst[ext=mp4]/18/worst',
                'outtmpl': str(video_path),
                'quiet': True,
                'no_warnings': True,
                'http_chunk_size': 10485760
            },
            # Strategy 3: Any available format
            {
                'format': 'best/worst',
                'outtmpl': str(video_path),
                'quiet': True,
                'no_warnings': True
            }
        ]
        
        actual_video_path = None
        
        for i, opts in enumerate(strategies):
            try:
                self.logger.info(f"🔄 Trying download method {i+1}/3...")
                
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    actual_video_path = ydl.prepare_filename(info)
                
                # Verify file exists and has size > 0
                if os.path.exists(actual_video_path) and os.path.getsize(actual_video_path) > 1000:
                    self.logger.info(f"✅ Download successful: {os.path.basename(actual_video_path)}")
                    break
                else:
                    actual_video_path = None
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Method {i+1} failed: {str(e)[:100]}...")
                actual_video_path = None
                time.sleep(1)
                continue
        
        if not actual_video_path:
            self.logger.error("❌ All download methods failed")
            return None, None
        
        # Extract audio
        return self._extract_audio_robust(actual_video_path, video_id)
    
    def _extract_audio_robust(self, video_path: str, video_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Robust audio extraction with error handling"""
        self.logger.info("🎵 Extracting audio...")
        
        audio_path = self.temp_dir / f"{video_id}_audio.wav"
        
        # Try different FFmpeg commands
        commands = [
            # Command 1: Standard extraction
            [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                str(audio_path), '-y'
            ],
            # Command 2: With error recovery
            [
                'ffmpeg', '-i', video_path,
                '-vn', '-ar', '16000', '-ac', '1', '-f', 'wav',
                str(audio_path), '-y'
            ],
            # Command 3: Basic extraction
            [
                'ffmpeg', '-i', video_path,
                '-vn', str(audio_path), '-y'
            ]
        ]
        
        for i, cmd in enumerate(commands):
            try:
                self.logger.info(f"🔧 Audio extraction method {i+1}/3...")
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=120,
                    check=True
                )
                
                # Verify audio file was created
                if audio_path.exists() and audio_path.stat().st_size > 1000:
                    self.logger.info(f"✅ Audio extracted successfully ({audio_path.stat().st_size // 1024} KB)")
                    return video_path, str(audio_path)
                    
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"⚠️ Audio method {i+1} failed: FFmpeg error")
                continue
            except subprocess.TimeoutExpired:
                self.logger.warning(f"⚠️ Audio method {i+1} timed out")
                continue
            except Exception as e:
                self.logger.warning(f"⚠️ Audio method {i+1} failed: {e}")
                continue
        
        self.logger.error("❌ All audio extraction methods failed")
        return None, None
    
    def transcribe_audio(self, audio_path: str) -> Optional[Dict]:
        """Transcribe audio using Whisper"""
        self.logger.info("🎤 Transcribing speech to text...")
        
        try:
            # Load and check audio
            audio = AudioSegment.from_wav(audio_path)
            duration_seconds = len(audio) / 1000
            self.logger.info(f"Audio duration: {duration_seconds:.1f} seconds")
            
            if duration_seconds > 600:  # 10 minutes
                self.logger.warning("⚠️ Long audio detected. This may take a while...")
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )
            
            segments_count = len(result.get('segments', []))
            detected_language = result.get('language', 'unknown')
            
            self.logger.info(f"✅ Transcription complete!")
            self.logger.info(f"   Language detected: {detected_language}")
            self.logger.info(f"   Speech segments: {segments_count}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Transcription failed: {e}")
            return None
    
    def translate_segments(self, segments: List[Dict], target_language: str) -> List[Dict]:
        """Translate all segments to target language"""
        self.logger.info(f"🌐 Translating to {target_language}...")
        
        translated_segments = []
        successful_translations = 0
        
        for i, segment in enumerate(segments):
            original_text = segment.get('text', '').strip()
            if not original_text:
                continue
            
            # Progress indicator
            if i % 10 == 0 or i == len(segments) - 1:
                progress = (i + 1) / len(segments) * 100
                self.logger.info(f"   Translating segment {i+1}/{len(segments)} ({progress:.0f}%)")
            
            try:
                # Translate text
                translator = GoogleTranslator(source='auto', target=target_language)
                translated_text = translator.translate(original_text)
                
                if translated_text and translated_text != original_text:
                    successful_translations += 1
                    translated_segments.append({
                        'original': original_text,
                        'translated': translated_text,
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0)
                    })
                else:
                    # Keep original if translation failed
                    translated_segments.append({
                        'original': original_text,
                        'translated': original_text,
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0)
                    })
                
            except Exception as e:
                self.logger.warning(f"⚠️ Translation failed for segment {i+1}: {str(e)[:50]}...")
                # Keep original text as fallback
                translated_segments.append({
                    'original': original_text,
                    'translated': original_text,
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0)
                })
        
        self.logger.info(f"✅ Translation complete! ({successful_translations}/{len(segments)} successful)")
        return translated_segments
    
    def create_translated_audio_fixed(self, translated_segments: List[Dict], target_language: str, video_id: str) -> Optional[str]:
        """FIXED: Create translated audio track with proper file handling"""
        self.logger.info("🎙️ Generating translated speech...")
        
        if not translated_segments:
            self.logger.error("❌ No segments to process")
            return None
        
        try:
            # Calculate total duration
            total_duration = max(seg['end'] for seg in translated_segments) * 1000  # Convert to ms
            final_audio = AudioSegment.silent(duration=int(total_duration))
            
            successful_segments = 0
            
            for i, segment in enumerate(translated_segments):
                translated_text = segment['translated'].strip()
                if not translated_text:
                    continue
                
                # Progress indicator
                if i % 5 == 0 or i == len(translated_segments) - 1:
                    progress = (i + 1) / len(translated_segments) * 100
                    self.logger.info(f"   Processing speech {i+1}/{len(translated_segments)} ({progress:.0f}%)")
                
                try:
                    # FIXED: Use absolute path and ensure directory exists
                    temp_speech_file = self.temp_dir / f"speech_{i:04d}.mp3"
                    
                    # Ensure temp directory exists
                    temp_speech_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Generate speech with error handling
                    try:
                        tts = gTTS(
                            text=translated_text,
                            lang=target_language,
                            slow=False
                        )
                        
                        # Save with absolute path
                        tts.save(str(temp_speech_file.resolve()))
                        
                        # Verify file was created
                        if not temp_speech_file.exists() or temp_speech_file.stat().st_size == 0:
                            self.logger.warning(f"⚠️ TTS file not created for segment {i+1}")
                            continue
                            
                    except Exception as tts_error:
                        self.logger.warning(f"⚠️ TTS generation failed for segment {i+1}: {tts_error}")
                        continue
                    
                    # Load generated speech
                    try:
                        speech_audio = AudioSegment.from_mp3(str(temp_speech_file))
                    except Exception as load_error:
                        self.logger.warning(f"⚠️ Failed to load TTS audio for segment {i+1}: {load_error}")
                        temp_speech_file.unlink(missing_ok=True)
                        continue
                    
                    # Calculate timing
                    start_ms = int(segment['start'] * 1000)
                    end_ms = int(segment['end'] * 1000)
                    segment_duration = end_ms - start_ms
                    
                    # Adjust speech timing
                    if len(speech_audio) > segment_duration and segment_duration > 0:
                        # Speed up if too long
                        speed_factor = len(speech_audio) / segment_duration
                        if speed_factor <= 3.0:  # Reasonable speed limit
                            speech_audio = speech_audio.speedup(playback_speed=speed_factor)
                        else:
                            # Cut if speed would be too extreme
                            speech_audio = speech_audio[:segment_duration]
                    
                    elif len(speech_audio) < segment_duration:
                        # Add silence if too short
                        silence_needed = segment_duration - len(speech_audio)
                        speech_audio = speech_audio + AudioSegment.silent(duration=silence_needed)
                    
                    # Add fade for smoother transitions
                    if len(speech_audio) > 100:
                        speech_audio = speech_audio.fade_in(50).fade_out(50)
                    
                    # Overlay onto final track
                    final_audio = final_audio.overlay(speech_audio, position=start_ms)
                    successful_segments += 1
                    
                    # Clean up temp file
                    temp_speech_file.unlink(missing_ok=True)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Speech processing failed for segment {i+1}: {str(e)[:100]}...")
                    continue
            
            if successful_segments == 0:
                self.logger.error("❌ No speech segments were generated successfully")
                return None
            
            # Export final audio with absolute path
            final_audio_path = self.temp_dir / f"{video_id}_final_audio.wav"
            final_audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            final_audio.export(str(final_audio_path.resolve()), format="wav")
            
            # Verify final audio was created
            if not final_audio_path.exists() or final_audio_path.stat().st_size == 0:
                self.logger.error("❌ Final audio file was not created")
                return None
            
            self.logger.info(f"✅ Translated audio created! ({successful_segments}/{len(translated_segments)} segments)")
            return str(final_audio_path.resolve())
            
        except Exception as e:
            self.logger.error(f"❌ Audio creation failed: {e}")
            return None
    
    def merge_video_audio(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Merge translated audio with original video"""
        self.logger.info("🎬 Creating final translated video...")
        
        try:
            subprocess.run([
                'ffmpeg', '-i', video_path, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                '-map', '0:v:0', '-map', '1:a:0',
                output_path, '-y'
            ], capture_output=True, check=True, timeout=300)
            
            # Verify output file
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                file_size = os.path.getsize(output_path) // (1024 * 1024)  # MB
                self.logger.info(f"✅ Final video created! ({file_size} MB)")
                return True
            else:
                self.logger.error("❌ Output file not created or too small")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Video merging failed: FFmpeg error")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Video merging timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Video merging failed: {e}")
            return False
    
    def _clean_temp_files(self, video_id: str):
        """Clean up temporary files"""
        patterns = [f"{video_id}*", f"*{video_id}*", "speech_*.mp3"]
        for pattern in patterns:
            for file_path in self.temp_dir.glob(pattern):
                try:
                    file_path.unlink()
                except:
                    pass
    
    def translate_video(self, url: str, target_language: str, video_name: str = None) -> Optional[str]:
        """Complete video translation pipeline"""
        start_time = time.time()
        
        # Generate video ID
        if video_name:
            video_id = f"{video_name}_{target_language}"
        else:
            video_id = f"video_{int(time.time())}_{target_language}"
        
        self.logger.info(f"🎬 Starting translation to {target_language}")
        self.logger.info(f"Video ID: {video_id}")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Download video and extract audio
            video_path, audio_path = self.download_video_robust(url, video_id)
            if not video_path or not audio_path:
                return None
            
            # Step 2: Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            if not transcription or not transcription.get('segments'):
                self.logger.error("❌ No speech segments found in audio")
                return None
            
            # Step 3: Translate segments
            translated_segments = self.translate_segments(
                transcription['segments'], 
                target_language
            )
            if not translated_segments:
                self.logger.error("❌ No segments translated successfully")
                return None
            
            # Step 4: Create translated audio (FIXED VERSION)
            translated_audio_path = self.create_translated_audio_fixed(
                translated_segments, 
                target_language, 
                video_id
            )
            if not translated_audio_path:
                return None
            
            # Step 5: Create final video
            output_video_path = self.output_dir / f"{video_id}.mp4"
            success = self.merge_video_audio(
                video_path, 
                translated_audio_path, 
                str(output_video_path)
            )
            
            if not success:
                return None
            
            # Cleanup temp files
            self._clean_temp_files(video_id)
            
            # Success!
            processing_time = time.time() - start_time
            self.logger.info("=" * 60)
            self.logger.info(f"🎉 Translation completed successfully!")
            self.logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
            self.logger.info(f"📁 Output file: {output_video_path}")
            self.logger.info(f"📊 File size: {os.path.getsize(output_video_path) // (1024*1024)} MB")
            
            return str(output_video_path)
            
        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Translation interrupted by user")
            self._clean_temp_files(video_id)
            return None
        except Exception as e:
            self.logger.error(f"❌ Translation failed: {e}")
            self._clean_temp_files(video_id)
            return None

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed YouTube Speech Translator')
    parser.add_argument('--url', required=True, help='YouTube video URL')
    parser.add_argument('--language', required=True, help='Target language code (hi, es, en, etc.)')
    parser.add_argument('--name', help='Custom name for output file')
    parser.add_argument('--output', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Validate language
    try:
        GoogleTranslator(source='en', target=args.language).translate('test')
    except Exception:
        print(f"❌ Invalid language code: {args.language}")
        print("Common codes: hi (Hindi), es (Spanish),en (English), de (German)")
        return 1
    
    # Create translator and process video
    translator = FixedYouTubeTranslator(output_dir=args.output)
    result = translator.translate_video(args.url, args.language, args.name)
    
    if result:
        print(f"\n🎊 Success! Your translated video is ready:")
        print(f"📁 {result}")
        return 0
    else:
        print(f"\n💥 Translation failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    # If run directly, use test mode
    if len(sys.argv) == 1:
        print("🎬 Fixed YouTube Speech Translator - Test Mode")
        print("=" * 60)
        
        # Test configuration
        test_url = input("Enter YouTube URL (or press Enter for English video): ").strip()
        if not test_url:
            test_url = "https://www.youtube.com/watch?v=MqAkmZgvBg4"  # English video
        
        target_lang = input("Enter target language (hi/es/en/de, default=hi): ").strip()
        if not target_lang:
            target_lang = "hi"
        
        print(f"\n🚀 Starting translation:")
        print(f"📺 Video: {test_url}")
        print(f"🌍 Language: {target_lang}")
        print()
        
        # Run translation
        translator = FixedYouTubeTranslator()
        result = translator.translate_video(test_url, target_lang, "test_video")
        
        if result:
            print(f"\n🎉 Test completed! Check your output folder:")
            print(f"📁 {result}")
        else:
            print(f"\n❌ Test failed. Check error messages above.")
    else:
        # Command line mode
        sys.exit(main())