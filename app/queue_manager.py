from queue import Queue
from threading import Thread, Event
import time
from datetime import datetime
from app import db

class ImageProcessor:
    def __init__(self, max_retries=3, queue_size=100):
        self.queue = Queue(maxsize=queue_size)
        self.processing_thread = None
        self.stop_event = Event()
        self.max_retries = max_retries
        self.current_processing = None
        
    def start(self):
        if not self.processing_thread or not self.processing_thread.is_alive():
            self.stop_event.clear()
            self.processing_thread = Thread(target=self._process_queue, daemon=True)
            self.processing_thread.start()
    
    def stop(self):
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join()
    
    def add_to_queue(self, image_id, predictor, image_path, image_type):
        self.queue.put({
            'image_id': image_id,
            'predictor': predictor,
            'image_path': image_path,
            'image_type': image_type,
            'retries': 0,
            'added_time': datetime.now()
        })
        
    def _process_queue(self):
        while not self.stop_event.is_set():
            try:
                if not self.queue.empty():
                    task = self.queue.get()
                    self.current_processing = task
                    
                    # Update processing status
                    db.images.update_one(
                        {'_id': task['image_id']},
                        {'$set': {
                            'processing_status': 'processing',
                            'queue_time': datetime.now() - task['added_time']
                        }}
                    )
                    
                    success = False
                    error_msg = None
                    
                    while not success and task['retries'] < self.max_retries:
                        try:
                            # Run prediction
                            result = task['predictor'].predict(
                                task['image_path'],
                                task['image_type']
                            )
                            
                            # Update analytics
                            if result['prediction'] == 'Damage Detected':
                                analytics_update = {
                                    'damage_type': task['image_type'],
                                    'confidence': result['confidence'],
                                    'detection_time': datetime.now(),
                                    'processing_time': (datetime.now() - task['added_time']).total_seconds(),
                                    'retries': task['retries'],
                                    'success': True
                                }
                            else:
                                analytics_update = {
                                    'damage_type': task['image_type'],
                                    'detection_time': datetime.now(),
                                    'processing_time': (datetime.now() - task['added_time']).total_seconds(),
                                    'retries': task['retries'],
                                    'success': True
                                }
                            
                            db.analytics.insert_one(analytics_update)
                            
                            # Update image record
                            db.images.update_one(
                                {'_id': task['image_id']},
                                {'$set': {
                                    'prediction_results': result,
                                    'processing_status': 'complete',
                                    'processing_time': (datetime.now() - task['added_time']).total_seconds(),
                                    'completed_at': datetime.now()
                                }}
                            )
                            
                            success = True
                            
                        except Exception as e:
                            task['retries'] += 1
                            error_msg = str(e)
                            time.sleep(1)  # Wait before retry
                    
                    if not success:
                        # Update failed status
                        db.images.update_one(
                            {'_id': task['image_id']},
                            {'$set': {
                                'processing_status': 'failed',
                                'error': error_msg,
                                'failed_at': datetime.now()
                            }}
                        )
                        
                        # Log failure in analytics
                        db.analytics.insert_one({
                            'damage_type': task['image_type'],
                            'detection_time': datetime.now(),
                            'processing_time': (datetime.now() - task['added_time']).total_seconds(),
                            'retries': task['retries'],
                            'success': False,
                            'error': error_msg
                        })
                    
                    self.current_processing = None
                    self.queue.task_done()
                else:
                    time.sleep(0.1)  # Prevent CPU spinning
                    
            except Exception as e:
                print(f"Queue processing error: {str(e)}")
                time.sleep(1)  # Wait before continuing
                
    def get_queue_status(self):
        return {
            'queue_size': self.queue.qsize(),
            'current_processing': self.current_processing,
            'is_active': self.processing_thread and self.processing_thread.is_alive()
        } 