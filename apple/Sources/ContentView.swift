import SwiftUI

struct ContentView: View {
    @State private var text = ""
    @State private var status = "مدل در حال آماده‌سازی است"
    @State private var isBusy = false
    @FocusState private var editorFocused: Bool

    private let burgundy = Color(red: 0.39, green: 0.055, blue: 0.11)
    private let ink = Color(red: 0.12, green: 0.095, blue: 0.08)
    private let paper = Color(red: 0.965, green: 0.945, blue: 0.905)

    var body: some View {
        ZStack {
            paper.ignoresSafeArea()
            Circle()
                .fill(burgundy.opacity(0.055))
                .frame(width: 390, height: 390)
                .offset(x: -180, y: -330)

            VStack(alignment: .trailing, spacing: 0) {
                Spacer(minLength: 34)

                HStack(alignment: .firstTextBaseline) {
                    Text("BOZORG · 1.5")
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                        .tracking(1.3)
                        .foregroundStyle(burgundy.opacity(0.64))
                    Spacer()
                    Text("گویا")
                        .font(.system(size: 52, weight: .bold, design: .rounded))
                        .foregroundStyle(ink)
                }

                Text("متن را بنویس؛ همه‌چیز همین‌جا و آفلاین خوانده می‌شود.")
                    .font(.system(size: 16))
                    .foregroundStyle(ink.opacity(0.56))
                    .padding(.top, 5)

                ZStack(alignment: .topTrailing) {
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .fill(Color.white.opacity(0.8))
                        .overlay {
                            RoundedRectangle(cornerRadius: 28, style: .continuous)
                                .stroke(burgundy.opacity(editorFocused ? 0.34 : 0.11), lineWidth: 1)
                        }

                    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        Text("مثلاً: امروز هوا چقدر دل‌انگیز است…")
                            .font(.system(size: 19))
                            .foregroundStyle(ink.opacity(0.3))
                            .padding(.horizontal, 23)
                            .padding(.vertical, 21)
                    }

                    TextEditor(text: $text)
                        .focused($editorFocused)
                        .scrollContentBackground(.hidden)
                        .font(.system(size: 22))
                        .foregroundStyle(ink)
                        .multilineTextAlignment(.trailing)
                        .padding(.horizontal, 17)
                        .padding(.vertical, 13)
                        .background(Color.clear)
                        .disabled(isBusy)
                }
                .frame(minHeight: 240)
                .padding(.top, 32)

                HStack(spacing: 9) {
                    if isBusy { ProgressView().tint(burgundy).controlSize(.small) }
                    Text(status)
                        .font(.system(size: 13))
                        .foregroundStyle(ink.opacity(0.5))
                        .lineLimit(2)
                    Spacer()
                }
                .frame(height: 44)
                .padding(.horizontal, 5)

                Button {
                    editorFocused = false
                    Task { await speak() }
                } label: {
                    ZStack {
                        RoundedRectangle(cornerRadius: 24, style: .continuous)
                            .fill(burgundy)
                            .shadow(color: burgundy.opacity(0.22), radius: 18, y: 10)
                        HStack(spacing: 12) {
                            if isBusy { ProgressView().tint(.white) }
                            Text("بگو")
                                .font(.system(size: 30, weight: .bold, design: .rounded))
                                .foregroundStyle(.white)
                        }
                    }
                    .frame(height: 78)
                }
                .buttonStyle(.plain)
                .disabled(isBusy || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .opacity(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.46 : 1)
                .accessibilityLabel("بگو")

                Spacer(minLength: 28)

                HStack(spacing: 7) {
                    ForEach(0..<11, id: \.self) { index in
                        Capsule()
                            .fill(burgundy.opacity(0.15 + Double(index % 3) * 0.06))
                            .frame(width: 3, height: CGFloat(8 + (index * 7) % 21))
                    }
                }
                .frame(maxWidth: .infinity)
                .accessibilityHidden(true)
                .padding(.bottom, 14)
            }
            .padding(.horizontal, 26)
            .frame(maxWidth: 680)
        }
        .preferredColorScheme(.light)
    }

    @MainActor
    private func speak() async {
        isBusy = true
        status = "در حال ساختن صدا…"
        defer { isBusy = false }
        // The concrete Core ML engine is attached only after the exported model
        // manifest passes source, graph, and device prediction validation.
        status = "بستهٔ Core ML هنوز به این ساخت متصل نشده است"
    }
}

#Preview { ContentView() }
