from app import app
import datetime

@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

if __name__ == '__main__':
    app.run(debug=True) 