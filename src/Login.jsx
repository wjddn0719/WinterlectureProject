import { useState } from 'react';
import { Trash2, LogIn } from 'lucide-react';

function Login({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: ''
  });

  const handleSubmit = async () => {
    if (isLogin) {
      // 로그인 API 호출
      console.log('로그인:', formData);
      // TODO: 로그인 API 연결
      // const response = await fetch('YOUR_API_URL/login', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ email: formData.email, password: formData.password })
      // });
      
      // 임시로 로그인 성공 처리
      onLogin();
    } else {
      // 회원가입 API 호출
      console.log('회원가입:', formData);
      // TODO: 회원가입 API 연결
      // const response = await fetch('YOUR_API_URL/register', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(formData)
      // });
      
      alert('회원가입이 완료되었습니다!');
      setIsLogin(true);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 flex items-center justify-center p-4">
      <div className="max-w-md w-full">
        {/* 헤더 */}
        <div className="text-center mb-8">
          <div className="inline-block p-3 bg-green-500 rounded-full mb-4">
            <Trash2 className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            분리수거 도우미
          </h1>
          <p className="text-gray-600 text-lg">
            올바른 분리수거를 위한 첫 걸음
          </p>
        </div>

        {/* 로그인/회원가입 카드 */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* 탭 전환 */}
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => setIsLogin(true)}
              className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                isLogin
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              로그인
            </button>
            <button
              onClick={() => setIsLogin(false)}
              className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                !isLogin
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              회원가입
            </button>
          </div>

          <div className="space-y-4">
            {/* 이름 입력 (회원가입시에만) */}
            {!isLogin && (
              <div>
                <label className="block text-gray-700 font-medium mb-2">
                  이름
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  onKeyPress={handleKeyPress}
                  placeholder="이름을 입력하세요"
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:border-green-500 focus:outline-none text-gray-800 placeholder-gray-400"
                />
              </div>
            )}

            {/* 이메일 입력 */}
            <div>
              <label className="block text-gray-700 font-medium mb-2">
                이메일
              </label>
              <input
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                onKeyPress={handleKeyPress}
                placeholder="example@email.com"
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:border-green-500 focus:outline-none text-gray-800 placeholder-gray-400"
              />
            </div>

            {/* 비밀번호 입력 */}
            <div>
              <label className="block text-gray-700 font-medium mb-2">
                비밀번호
              </label>
              <input
                type="password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                onKeyPress={handleKeyPress}
                placeholder="비밀번호를 입력하세요"
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:border-green-500 focus:outline-none text-gray-800 placeholder-gray-400"
              />
            </div>

            {/* 로그인/회원가입 버튼 */}
            <button
              onClick={handleSubmit}
              className="w-full bg-green-500 text-white py-4 rounded-xl font-medium hover:bg-green-600 transition-colors flex items-center justify-center gap-2 mt-6"
            >
              <LogIn className="w-5 h-5" />
              {isLogin ? '로그인' : '회원가입'}
            </button>
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

export default Login;