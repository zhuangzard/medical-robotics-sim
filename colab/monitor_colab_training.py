#!/usr/bin/env python3
"""
Monitor Colab training progress and send updates to Telegram
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

def send_telegram(message: str, chat_id: str = "5862095637"):
    """Send message to Telegram via OpenClaw"""
    # Use OpenClaw's message tool via CLI
    cmd = [
        'openclaw', 'message', 'send',
        '--channel', 'telegram',
        '--target', chat_id,
        '--message', message
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except:
        return False

def check_progress():
    """Check training progress from Drive file"""
    # Check common Drive sync locations
    possible_paths = [
        Path.home() / "Google Drive" / "medical-robotics-progress" / "training_progress.json",
        Path.home() / "GoogleDrive" / "medical-robotics-progress" / "training_progress.json",
        Path.home() / "Library/CloudStorage/GoogleDrive-*/My Drive/medical-robotics-progress/training_progress.json"
    ]
    
    for path_pattern in possible_paths:
        # Handle wildcards
        if '*' in str(path_pattern):
            import glob
            matches = glob.glob(str(path_pattern))
            if matches:
                path = Path(matches[0])
                if path.exists():
                    break
        else:
            if path_pattern.exists():
                path = path_pattern
                break
    else:
        return None
    
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except:
        return None

def format_progress_message(progress: dict) -> str:
    """Format progress data as Telegram message"""
    status = progress.get('status', 'unknown')
    timestamp = progress.get('timestamp', 'N/A')
    gpu = progress.get('gpu', 'Unknown')
    message_text = progress.get('message', '')
    
    # Format timestamp
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        time_str = dt.strftime('%H:%M:%S')
    except:
        time_str = timestamp
    
    # Status emoji
    status_emoji = {
        'started': '🏁',
        'training': '🔄',
        'complete': '✅',
        'saved': '💾',
        'error': '❌'
    }.get(status, '📊')
    
    msg = f"{status_emoji} **Colab Training Status**\n\n"
    msg += f"**Status**: {status.upper()}\n"
    msg += f"**Time**: {time_str}\n"
    msg += f"**GPU**: {gpu}\n"
    
    if message_text:
        msg += f"**Message**: {message_text}\n"
    
    # Add ETA if training
    if status == 'training' and 'eta_hours' in progress:
        eta = progress['eta_hours']
        msg += f"**ETA**: ~{eta} hours\n"
    
    # Add duration if complete
    if status in ['complete', 'saved'] and 'duration_hours' in progress:
        duration = progress['duration_hours']
        msg += f"**Duration**: {duration:.1f} hours\n"
    
    # Add results path if saved
    if status == 'saved' and 'path' in progress:
        path = progress['path']
        msg += f"\n📂 **Results**: `{path}`"
    
    return msg

def monitor_training(check_interval: int = 600, max_runtime: int = 86400):
    """
    Monitor training progress
    
    Args:
        check_interval: Seconds between checks (default: 600 = 10 min)
        max_runtime: Maximum monitoring time in seconds (default: 86400 = 24h)
    """
    print(f"🔍 Starting Colab training monitor")
    print(f"   Check interval: {check_interval}s ({check_interval/60:.0f} min)")
    print(f"   Max runtime: {max_runtime}s ({max_runtime/3600:.0f} hours)")
    print()
    
    start_time = time.time()
    last_status = None
    last_message_time = 0
    
    # Send initial message
    send_telegram("🚀 **Colab 训练监控已启动**\n\n每 10 分钟检查一次进度...")
    
    while True:
        current_time = time.time()
        
        # Check if max runtime exceeded
        if current_time - start_time > max_runtime:
            print(f"⏰ Max runtime ({max_runtime/3600:.0f}h) exceeded, stopping monitor")
            send_telegram("⏰ **监控已停止** (达到最大运行时间 24h)")
            break
        
        # Check progress
        progress = check_progress()
        
        if progress:
            status = progress.get('status')
            
            # Send update if status changed or every hour
            should_send = (
                status != last_status or 
                (current_time - last_message_time) > 3600
            )
            
            if should_send:
                msg = format_progress_message(progress)
                if send_telegram(msg):
                    print(f"✅ Sent update: {status}")
                    last_message_time = current_time
                else:
                    print(f"❌ Failed to send update")
                
                last_status = status
            
            # Check if training complete
            if status in ['complete', 'saved', 'error']:
                print(f"🎉 Training finished with status: {status}")
                
                # Send final message
                final_msg = format_progress_message(progress)
                final_msg += "\n\n🎉 **监控完成！**"
                send_telegram(final_msg)
                break
        else:
            # No progress file found
            if last_status is None:
                # First check, file might not exist yet
                print("⏳ Waiting for progress file to appear...")
            else:
                print("⚠️  Progress file not found (Drive sync issue?)")
        
        # Wait before next check
        print(f"💤 Sleeping {check_interval}s...")
        time.sleep(check_interval)
    
    print("✅ Monitor stopped")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Colab training')
    parser.add_argument('--interval', type=int, default=600,
                       help='Check interval in seconds (default: 600 = 10 min)')
    parser.add_argument('--max-hours', type=int, default=24,
                       help='Maximum monitoring time in hours (default: 24)')
    
    args = parser.parse_args()
    
    max_runtime = args.max_hours * 3600
    
    try:
        monitor_training(check_interval=args.interval, max_runtime=max_runtime)
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user")
        send_telegram("🛑 **Colab 监控已手动停止**")
