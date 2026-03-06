# 使用示例大全

## 🎯 基础使用示例

### 1. 简单文本检测

#### Python客户端示例
```python
import requests
import json

# 二分类检测
def binary_detection(text):
    url = "http://localhost:8000/detect"
    payload = {
        "text": text,
        "return_probability": True
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    if result["success"]:
        data = result["data"]
        print(f"文本: {data['text']}")
        print(f"是否违规: {'是' if data['is_forbidden'] else '否'}")
        print(f"置信度: {data['probability']:.4f}")
        return data['is_forbidden']
    else:
        print(f"检测失败: {result['message']}")
        return None

# 多分类检测
def multiclass_detection(text):
    url = "http://localhost:8001/detect"  # 多分类通常使用不同端口
    payload = {
        "text": text,
        "return_probabilities": True
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    if result["success"]:
        data = result["data"]
        print(f"文本: {data['text']}")
        print(f"预测类别: {data['predicted_category']}")
        print(f"置信度: {data['confidence']:.4f}")
        print("各类别概率:")
        for category, prob in data['probabilities'].items():
            print(f"  {category}: {prob:.4f}")
        return data['predicted_category']
    else:
        print(f"检测失败: {result['message']}")
        return None

# 使用示例
if __name__ == "__main__":
    test_texts = [
        "这是一篇很好的文章",
        "我要杀了你",
        "色情相关内容"
    ]
    
    for text in test_texts:
        print("=" * 50)
        print(f"检测文本: {text}")
        binary_result = binary_detection(text)
        multiclass_result = multiclass_detection(text)
        print()
```

#### 命令行示例
```bash
# 二分类检测
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{"text": "测试文本", "return_probability": true}'

# 多分类检测
curl -X POST "http://localhost:8001/detect" \
     -H "Content-Type: application/json" \
     -d '{"text": "我要杀了你", "return_probabilities": true}'

# 批量检测
curl -X POST "http://localhost:8000/batch_detect" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["文本1", "文本2", "文本3"]}'
```

### 2. 批量处理示例

```python
import requests
import time
from concurrent.futures import ThreadPoolExecutor

class BatchDetector:
    def __init__(self, api_url="http://localhost:8000", batch_size=32):
        self.api_url = api_url
        self.batch_size = batch_size
    
    def detect_batch(self, texts):
        """批量检测文本"""
        url = f"{self.api_url}/batch_detect"
        payload = {
            "texts": texts,
            "batch_size": self.batch_size
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            result = response.json()
            
            if result["success"]:
                return result["data"]
            else:
                print(f"批量检测失败: {result['message']}")
                return []
        except Exception as e:
            print(f"请求异常: {e}")
            return []
    
    def detect_parallel(self, texts, max_workers=4):
        """并行批量检测"""
        # 将文本分批
        batches = [texts[i:i + self.batch_size] 
                  for i in range(0, len(texts), self.batch_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.detect_batch, batch) 
                      for batch in batches]
            
            for future in futures:
                batch_results = future.result()
                results.extend(batch_results)
        
        return results

# 使用示例
def main():
    # 准备测试数据
    test_texts = [
        f"这是第{i}条评论内容" for i in range(1000)
    ]
    
    # 添加一些违规内容用于测试
    test_texts.extend([
        "你真是个垃圾",
        "我要杀了你", 
        "色情内容展示"
    ])
    
    detector = BatchDetector(api_url="http://localhost:8000")
    
    # 性能测试
    start_time = time.time()
    results = detector.detect_parallel(test_texts, max_workers=4)
    end_time = time.time()
    
    print(f"处理 {len(test_texts)} 条文本耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每秒处理: {len(test_texts)/(end_time - start_time):.2f} 条")
    
    # 统计结果
    violations = [r for r in results if r.get('is_forbidden', False)]
    print(f"发现违规内容: {len(violations)} 条")

if __name__ == "__main__":
    main()
```

## 🏢 企业级应用示例

### 1. 博客评论审核系统

```python
import requests
from datetime import datetime
from typing import List, Dict, Optional
import logging

class CommentModerator:
    def __init__(self, api_url: str, threshold: float = 0.7):
        self.api_url = api_url
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # 定义不同类型的处理策略
        self.action_rules = {
            'normal': 'approve',
            'insult': 'warn_or_reject',
            'threat': 'immediate_reject',
            'porn': 'immediate_reject', 
            'fraud': 'flag_for_review'
        }
    
    def moderate_comment(self, comment: str, user_id: str) -> Dict:
        """审核单条评论"""
        try:
            # 调用多分类检测
            result = self._call_detection_api(comment)
            
            if not result['success']:
                raise Exception(f"检测API调用失败: {result['message']}")
            
            detection_data = result['data']
            predicted_category = detection_data['predicted_category']
            confidence = detection_data['confidence']
            
            # 根据类别和置信度决定处理动作
            action = self._determine_action(predicted_category, confidence)
            
            moderation_result = {
                'comment': comment,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'category': predicted_category,
                'confidence': confidence,
                'action': action,
                'details': detection_data
            }
            
            self.logger.info(f"评论审核完成: {moderation_result}")
            return moderation_result
            
        except Exception as e:
            self.logger.error(f"评论审核失败: {e}")
            return {
                'comment': comment,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'action': 'manual_review'
            }
    
    def _call_detection_api(self, text: str) -> Dict:
        """调用检测API"""
        url = f"{self.api_url}/detect"
        payload = {
            "text": text,
            "return_probabilities": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        return response.json()
    
    def _determine_action(self, category: str, confidence: float) -> str:
        """根据类别和置信度确定处理动作"""
        base_action = self.action_rules.get(category, 'manual_review')
        
        # 根据置信度调整处理严格程度
        if confidence < 0.5:
            return 'approve'  # 低置信度，倾向于通过
        elif confidence > 0.9:
            if category in ['threat', 'porn']:
                return 'immediate_reject'  # 高置信度违规，立即拒绝
            elif category == 'insult':
                return 'warn_or_reject'  # 高置信度辱骂，警告或拒绝
        
        return base_action

# 使用示例
def blog_comment_system():
    moderator = CommentModerator("http://localhost:8001")
    
    sample_comments = [
        ("user123", "这篇文章写得真好，学到了很多"),
        ("user456", "你这个白痴，什么都不懂"),
        ("user789", "我要让你付出代价"),
        ("user101", "免费看片网站：xxx.com"),
        ("user202", "刷单兼职，日赚300-500元")
    ]
    
    for user_id, comment in sample_comments:
        result = moderator.moderate_comment(comment, user_id)
        print(f"用户 {user_id}: {result['action']} - {result['category']} ({result['confidence']:.2f})")
        print(f"评论: {comment}\n")
```

### 2. 社交媒体内容过滤

```python
import asyncio
import aiohttp
from typing import List, Dict
import time

class SocialMediaFilter:
    def __init__(self, api_base_url: str, rate_limit: int = 100):
        self.api_base_url = api_base_url
        self.rate_limit = rate_limit  # 每分钟最大请求数
        self.request_count = 0
        self.last_reset = time.time()
    
    async def filter_posts(self, posts: List[Dict]) -> List[Dict]:
        """异步过滤社交媒体帖子"""
        filtered_posts = []
        
        # 检查速率限制
        current_time = time.time()
        if current_time - self.last_reset > 60:
            self.request_count = 0
            self.last_reset = current_time
        
        if self.request_count >= self.rate_limit:
            wait_time = 60 - (current_time - self.last_reset)
            print(f"达到速率限制，等待 {wait_time:.1f} 秒")
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.last_reset = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for post in posts:
                task = self._filter_single_post(session, post)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            filtered_posts = [post for post in results if post is not None]
        
        self.request_count += len(posts)
        return filtered_posts
    
    async def _filter_single_post(self, session: aiohttp.ClientSession, post: Dict) -> Optional[Dict]:
        """过滤单个帖子"""
        try:
            # 检测标题
            title_result = await self._detect_content(session, post.get('title', ''))
            if title_result and title_result['is_violation']:
                post['violation_reason'] = f"标题违规: {title_result['category']}"
                return post
            
            # 检测正文
            content_result = await self._detect_content(session, post.get('content', ''))
            if content_result and content_result['is_violation']:
                post['violation_reason'] = f"内容违规: {content_result['category']}"
                return post
            
            # 检测标签
            tags_result = await self._detect_tags(session, post.get('tags', []))
            if tags_result and tags_result['is_violation']:
                post['violation_reason'] = f"标签违规: {tags_result['category']}"
                return post
            
            # 无违规内容
            post['status'] = 'approved'
            return post
            
        except Exception as e:
            print(f"帖子过滤出错: {e}")
            post['status'] = 'error'
            post['error_message'] = str(e)
            return post
    
    async def _detect_content(self, session: aiohttp.ClientSession, text: str) -> Optional[Dict]:
        """检测文本内容"""
        if not text.strip():
            return None
            
        url = f"{self.api_base_url}/detect"
        payload = {
            "text": text,
            "return_probabilities": True
        }
        
        try:
            async with session.post(url, json=payload, timeout=10) as response:
                if response.status == 200:
                    result = await response.json()
                    if result['success']:
                        data = result['data']
                        return {
                            'is_violation': data['predicted_category'] != 'normal',
                            'category': data['predicted_category'],
                            'confidence': data['confidence']
                        }
        except Exception as e:
            print(f"内容检测失败: {e}")
        
        return None
    
    async def _detect_tags(self, session: aiohttp.ClientSession, tags: List[str]) -> Optional[Dict]:
        """检测标签"""
        if not tags:
            return None
            
        # 将标签组合成文本进行检测
        tags_text = " ".join(tags)
        return await self._detect_content(session, tags_text)

# 使用示例
async def social_media_example():
    filter_system = SocialMediaFilter("http://localhost:8001")
    
    sample_posts = [
        {
            'id': 1,
            'title': '今日分享',
            'content': '今天天气真好，出去散步了',
            'tags': ['生活', '日常']
        },
        {
            'id': 2, 
            'title': '愤怒声明',
            'content': '我要杀了那个混蛋',
            'tags': ['愤怒', '威胁']
        },
        {
            'id': 3,
            'title': '技术分享',
            'content': 'Python编程技巧分享给大家',
            'tags': ['技术', '编程']
        }
    ]
    
    start_time = time.time()
    filtered_posts = await filter_system.filter_posts(sample_posts)
    end_time = time.time()
    
    print(f"处理 {len(sample_posts)} 个帖子耗时: {end_time - start_time:.2f} 秒")
    
    for post in filtered_posts:
        status = post.get('status', 'unknown')
        reason = post.get('violation_reason', '无')
        print(f"帖子 {post['id']}: {status} - {reason}")

# 运行示例
if __name__ == "__main__":
    asyncio.run(social_media_example())
```

## 🛡️ 安全加固示例

### 1. 带认证的API客户端

```python
import requests
import hashlib
import hmac
import time
from typing import Optional

class SecureAPIClient:
    def __init__(self, base_url: str, api_key: str, secret_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.secret_key = secret_key
        self.session = requests.Session()
    
    def _generate_signature(self, timestamp: str, body: str = "") -> str:
        """生成请求签名"""
        message = f"{timestamp}{body}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_secure_request(self, endpoint: str, data: dict = None) -> dict:
        """发送安全请求"""
        url = f"{self.base_url}{endpoint}"
        timestamp = str(int(time.time()))
        
        headers = {
            'X-API-Key': self.api_key,
            'X-Timestamp': timestamp,
            'Content-Type': 'application/json'
        }
        
        body = ""
        if data:
            import json
            body = json.dumps(data, separators=(',', ':'))
            headers['X-Signature'] = self._generate_signature(timestamp, body)
        
        try:
            if data:
                response = self.session.post(url, headers=headers, data=body, timeout=30)
            else:
                response = self.session.get(url, headers=headers, timeout=30)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'message': f'请求失败: {str(e)}'
            }
    
    def detect_content(self, text: str, return_probs: bool = True) -> dict:
        """安全的内容检测"""
        data = {
            'text': text,
            'return_probabilities': return_probs
        }
        return self._make_secure_request('/detect', data)
    
    def batch_detect(self, texts: list, batch_size: int = 32) -> dict:
        """安全的批量检测"""
        data = {
            'texts': texts,
            'batch_size': batch_size
        }
        return self._make_secure_request('/batch_detect', data)
    
    def get_health(self) -> dict:
        """获取服务健康状态"""
        return self._make_secure_request('/health')

# 使用示例
def secure_detection_example():
    client = SecureAPIClient(
        base_url="http://localhost:8001",
        api_key="your-api-key",
        secret_key="your-secret-key"
    )
    
    # 检测文本
    result = client.detect_content("测试内容")
    print("检测结果:", result)
    
    # 健康检查
    health = client.get_health()
    print("服务状态:", health)

if __name__ == "__main__":
    secure_detection_example()
```
