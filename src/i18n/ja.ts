export const ja = {
  welcome: {
    title: 'ジョージア フォトブース',
    subtitle: 'ジョージアの思い出をあなたに',
    description: 'ジョージアの美しい風景をバックに\n記念写真を撮りましょう！',
    start: 'スタート',
  },
  capture: {
    title: '写真を撮影・選択',
    takePhoto: 'カメラで撮影',
    uploadPhoto: 'アルバムから選択',
    retake: '撮り直す',
    usePhoto: 'この写真を使う',
  },
  background: {
    title: '背景を選んでください',
    processing: '背景を処理中...',
    tbilisi: 'トビリシ旧市街',
    kutaisi: 'プロメテウス洞窟（クタイシ）',
    vardzia: 'ヴァルジア洞窟修道院',
    uplistsikhe: 'ウプリスツィヘ古代都市',
    tskaltubo: 'ツカルトゥボ鉱泉',
    next: '次へ',
  },
  preview: {
    title: 'プレビュー',
    dragToMove: 'ドラッグで位置調整',
    pinchToZoom: 'ピンチで拡大・縮小',
    changeBackground: '背景を変更',
    complete: '完成！',
  },
  download: {
    title: '記念写真の完成！',
    downloadPhoto: '写真をダウンロード',
    takeAnother: 'もう一枚撮る',
    thankYou: 'ご利用ありがとうございました\nまたのお越しをお待ちしています',
  },
  common: {
    loading: '読み込み中...',
    error: 'エラーが発生しました',
    back: '戻る',
    step: 'ステップ',
  },
}

export interface Translations {
  welcome: { title: string; subtitle: string; description: string; start: string }
  capture: { title: string; takePhoto: string; uploadPhoto: string; retake: string; usePhoto: string }
  background: { title: string; processing: string; tbilisi: string; kutaisi: string; vardzia: string; uplistsikhe: string; tskaltubo: string; next: string }
  preview: { title: string; dragToMove: string; pinchToZoom: string; changeBackground: string; complete: string }
  download: { title: string; downloadPhoto: string; takeAnother: string; thankYou: string }
  common: { loading: string; error: string; back: string; step: string }
}
