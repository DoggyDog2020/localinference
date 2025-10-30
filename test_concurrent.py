#!/usr/bin/env python3
"""
Test script to simulate 20 concurrent users making requests to the server.
"""
import asyncio
import aiohttp
import time
from datetime import datetime

async def make_chat_request(session, user_id):
    """Make a chat request and measure response time."""
    start = time.time()
    try:
        async with session.post(
            'http://localhost:8000/chat',
            json={'text': f'User {user_id}: Tell me a fun fact about space'},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            result = await response.json()
            elapsed = time.time() - start
            print(f"✓ User {user_id:2d} completed in {elapsed:.2f}s")
            return {'user': user_id, 'time': elapsed, 'success': True, 'response': result}
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ User {user_id:2d} failed after {elapsed:.2f}s: {e}")
        return {'user': user_id, 'time': elapsed, 'success': False, 'error': str(e)}

async def make_summary_request(session, user_id):
    """Make a summarization request and measure response time."""
    start = time.time()
    text_to_summarize = """
    Artificial intelligence is transforming the world in unprecedented ways.
    From healthcare to transportation, AI systems are being deployed to solve
    complex problems. Machine learning algorithms can now diagnose diseases,
    drive cars autonomously, and even create art. However, these advances also
    raise important ethical questions about privacy, bias, and the future of work.
    """
    try:
        async with session.post(
            'http://localhost:8000/summarize',
            json={'text': text_to_summarize},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            result = await response.json()
            elapsed = time.time() - start
            print(f"✓ User {user_id:2d} completed in {elapsed:.2f}s")
            return {'user': user_id, 'time': elapsed, 'success': True, 'response': result}
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ User {user_id:2d} failed after {elapsed:.2f}s: {e}")
        return {'user': user_id, 'time': elapsed, 'success': False, 'error': str(e)}

async def run_concurrent_test(num_users=20, test_type='chat'):
    """Run concurrent test with specified number of users."""
    print(f"\n{'='*60}")
    print(f"Testing {test_type.upper()} with {num_users} concurrent users")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        # Create all tasks
        if test_type == 'chat':
            tasks = [make_chat_request(session, i+1) for i in range(num_users)]
        else:
            tasks = [make_summary_request(session, i+1) for i in range(num_users)]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    # Calculate statistics
    successful = sum(1 for r in results if r['success'])
    failed = num_users - successful
    avg_time = sum(r['time'] for r in results if r['success']) / max(successful, 1)
    max_time = max(r['time'] for r in results)
    min_time = min(r['time'] for r in results)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total time:        {total_time:.2f}s")
    print(f"Successful:        {successful}/{num_users}")
    print(f"Failed:            {failed}/{num_users}")
    print(f"Average time:      {avg_time:.2f}s")
    print(f"Min time:          {min_time:.2f}s")
    print(f"Max time:          {max_time:.2f}s")
    print(f"{'='*60}\n")

    return results

async def main():
    """Main test runner."""
    print("\n🚀 Starting Concurrent Load Test")
    print("This will test the server with 20 concurrent users\n")

    # Test chat endpoint with 20 users
    chat_results = await run_concurrent_test(num_users=20, test_type='chat')

    # Wait a bit between tests
    print("Waiting 5 seconds before next test...\n")
    await asyncio.sleep(5)

    # Test summarization endpoint with 20 users
    summary_results = await run_concurrent_test(num_users=20, test_type='summarize')

    print("\n✅ All tests completed!")
    print(f"\n💡 Server Configuration:")
    print(f"   - ThreadPoolExecutor: 20 workers")
    print(f"   - Semaphore limit: 15 concurrent GPU requests")
    print(f"   - GPU: MPS (Metal Performance Shaders) on M4 Mac")

if __name__ == "__main__":
    asyncio.run(main())
