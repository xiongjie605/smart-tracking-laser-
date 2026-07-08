#!/bin/bash
# OpenClaw 一键配置脚本 / OpenClaw One-Click Setup Script
# 用于配置阿里云百炼 API Key / For configuring Alibaba Cloud Bailian API Key

set -e

CONFIG_DIR="$HOME/.openclaw"
ENV_FILE="$CONFIG_DIR/.env"
OPENCLAW_JSON="$CONFIG_DIR/openclaw.json"

echo "========================================"
echo "  OpenClaw 配置向导 / Setup Wizard"
echo "========================================"
echo ""

# 检查是否已有配置
if [ -f "$ENV_FILE" ] && grep -q "DASHSCOPE_API_KEY=sk-" "$ENV_FILE" 2>/dev/null; then
    echo "✓ 检测到已有百炼 API Key 配置 / Detected existing Bailian API Key configuration"
    echo ""
    read -p "是否重新配置？(y/N) / Reconfigure? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "配置已取消 / Configuration cancelled"
        exit 0
    fi
fi

echo "请输入阿里云百炼 API Key："
echo "(获取地址: https://bailian.console.aliyun.com/)"
echo ""
echo "Please enter Alibaba Cloud Bailian API Key:"
echo "(Get it from: https://bailian.console.aliyun.com/)"
echo ""
read -p "DASHSCOPE_API_KEY: " API_KEY

if [ -z "$API_KEY" ]; then
    echo "错误：API Key 不能为空 / Error: API Key cannot be empty"
    exit 1
fi

if [[ ! "$API_KEY" == sk-* ]]; then
    echo "警告：API Key 通常以 'sk-' 开头，请确认是否正确 / Warning: API Key usually starts with 'sk-', please verify"
fi

# 创建 .env 文件
mkdir -p "$CONFIG_DIR"
cat > "$ENV_FILE" << EOF
# 阿里云百炼 API Key / Alibaba Cloud Bailian API Key
DASHSCOPE_API_KEY=$API_KEY
EOF

echo ""
echo "✓ 百炼 API Key 已保存到 $ENV_FILE"
echo "✓ Bailian API Key saved to $ENV_FILE"
echo ""

# 更新 openclaw.json 中的 bailian apiKey
if [ -f "$OPENCLAW_JSON" ]; then
    # 使用 python 更新 JSON
    python3 << EOF
import json
import sys
from pathlib import Path

try:
    p = Path("$OPENCLAW_JSON")
    config = json.loads(p.read_text(encoding="utf-8-sig"))
    
    # 更新 bailian apiKey / Update bailian apiKey
    if "models" in config and "providers" in config["models"]:
        if "bailian" in config["models"]["providers"]:
            config["models"]["providers"]["bailian"]["apiKey"] = "$API_KEY"
    
    p.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    print("✓ 已更新 openclaw.json 中的百炼配置")
    print("✓ Updated Bailian configuration in openclaw.json")
except Exception as e:
    print(f"警告：更新 openclaw.json 失败: {e}", file=sys.stderr)
    print(f"Warning: Failed to update openclaw.json: {e}", file=sys.stderr)
EOF
fi

echo ""
echo "========================================"
echo "  配置完成！ / Configuration Complete!"
echo "========================================"
echo ""
echo "已配置的功能 / Configured Features:"
echo "  ✓ 图像生成/编辑 / Image generation/editing"
echo "  ✓ 语音合成（CosyVoice） / Voice synthesis (CosyVoice)"
echo "  ✓ 语音识别 / Speech recognition"
echo "  ✓ 视频生成 / Video generation"
echo "  ✓ PPT演示生成 / PPT presentation generation"
echo ""
echo "可选配置（通过和 OpenClaw 对话完成） / Optional Configuration (via OpenClaw chat):"
echo "  • QQBot - 发送 QQ 消息 / Send QQ messages"
echo "  • 其他模型 API Key / Other model API Keys"
echo ""
echo "现在可以开始使用 OpenClaw 了！"
echo "You can now start using OpenClaw!"
echo ""
