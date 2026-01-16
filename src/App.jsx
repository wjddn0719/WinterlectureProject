import { useState } from 'react';
import { Trash2, Send, Upload, X } from 'lucide-react';

function App() {
  const [question, setQuestion] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const removeImage = () => {
    setImage(null);
    setImagePreview(null);
  };

  const handleSubmit = () => {
    // 나중에 백엔드 연결 부분
    console.log('질문:', question);
    console.log('이미지:', image);
    // TODO: API 호출 로직 추가
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && (question || image)) {
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* 헤더 */}
        <div className="text-center mb-8">
          <div className="inline-block p-3 bg-green-500 rounded-full mb-4">
            <Trash2 className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            분리수거 도우미
          </h1>
          <p className="text-gray-600 text-lg">
            사진을 올리거나 질문을 입력하면 올바른 분리수거 방법을 알려드려요
          </p>
        </div>

        {/* 메인 카드 */}
        <div className="bg-white rounded-2xl shadow-xl p-12">
          <div className="space-y-10">
            {/* 이미지 업로드 영역 */}
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-12 hover:border-green-500 transition-colors">
              {imagePreview ? (
                <div className="relative">
                  <img 
                    src={imagePreview} 
                    alt="업로드된 이미지" 
                    className="w-full h-80 object-contain rounded-lg"
                  />
                  <button
                    onClick={removeImage}
                    className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              ) : (
                <label className="flex flex-col items-center cursor-pointer">
                  <Upload className="w-12 h-12 text-gray-400 mb-3" />
                  <span className="text-gray-600 font-medium mb-1">
                    사진 업로드하기
                  </span>
                  <span className="text-gray-400 text-sm">
                    분리수거가 궁금한 물건을 촬영해주세요
                  </span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </label>
              )}
            </div>

            {/* 질문 입력 영역 */}
            <div className="space-y-3">
              <label className="block text-gray-700 font-medium text-lg">
                질문하기
              </label>
              <div className="relative">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="예: 피자 박스는 어떻게 버리나요?"
                  className="w-full px-5 py-5 pr-14 border-2 border-gray-300 rounded-xl focus:border-green-500 focus:outline-none text-gray-800 placeholder-gray-400 text-lg"
                />
                <button
                  onClick={handleSubmit}
                  className="absolute right-2 top-1/2 -translate-y-1/2 bg-green-500 text-white p-2 rounded-lg hover:bg-green-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
                  disabled={!question && !image}
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </div>


          </div>
        </div>

        {/* 푸터 */}
        <div className="text-center mt-6 text-gray-600 text-sm">
          <p>국가환경교육센터 재활용품 분리배출 가이드라인 기반</p>
        </div>
      </div>
    </div>
  );
}

export default App;