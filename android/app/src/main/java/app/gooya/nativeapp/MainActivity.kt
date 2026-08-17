package app.gooya.nativeapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

private val Burgundy = Color(0xFF630E1C)
private val Paper = Color(0xFFF6F1E7)
private val Ink = Color(0xFF1F1814)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val readiness = LiteRtGooyaEngine(this).readiness()
        setContent { MaterialTheme { GooyaScreen(readiness) } }
    }
}

@Composable
private fun GooyaScreen(readiness: LiteRtGooyaEngine.Readiness) {
    var text by remember { mutableStateOf("") }
    var status by remember {
        mutableStateOf(
            when (readiness) {
                LiteRtGooyaEngine.Readiness.Ready -> "مدل آماده است"
                is LiteRtGooyaEngine.Readiness.Missing -> "بستهٔ LiteRT هنوز نصب نشده است"
            }
        )
    }
    Surface(color = Paper, modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.fillMaxSize().padding(horizontal = 24.dp, vertical = 32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("BOZORG · 1.5", color = Burgundy.copy(alpha = .68f), fontSize = 11.sp)
                Text("گویا", color = Ink, fontSize = 48.sp, fontWeight = FontWeight.Bold)
            }
            Text(
                "متن را بنویس؛ همه‌چیز همین‌جا و آفلاین خوانده می‌شود.",
                modifier = Modifier.fillMaxWidth(),
                color = Ink.copy(alpha = .58f),
                fontSize = 16.sp,
                textAlign = TextAlign.End,
            )
            Spacer(Modifier.height(28.dp))
            BasicTextField(
                value = text,
                onValueChange = { text = it },
                modifier = Modifier.fillMaxWidth().height(240.dp)
                    .background(Color.White.copy(alpha = .85f), RoundedCornerShape(28.dp))
                    .padding(20.dp),
                textStyle = androidx.compose.ui.text.TextStyle(
                    color = Ink, fontSize = 22.sp, textAlign = TextAlign.End
                ),
                decorationBox = { inner ->
                    if (text.isEmpty()) Text(
                        "مثلاً: امروز هوا چقدر دل‌انگیز است…",
                        modifier = Modifier.fillMaxWidth(),
                        color = Ink.copy(alpha = .35f),
                        fontSize = 20.sp,
                        textAlign = TextAlign.End,
                    )
                    inner()
                },
            )
            Spacer(Modifier.height(12.dp))
            Text(status, color = Ink.copy(alpha = .52f), fontSize = 13.sp)
            Spacer(Modifier.height(14.dp))
            Button(
                onClick = {
                    status = when (readiness) {
                        LiteRtGooyaEngine.Readiness.Ready -> "موتور صوت هنوز به رابط وصل نشده است"
                        is LiteRtGooyaEngine.Readiness.Missing ->
                            "بستهٔ LiteRT ناقص است (${readiness.assets.size} فایل)"
                    }
                },
                enabled = text.isNotBlank(),
                modifier = Modifier.fillMaxWidth().height(78.dp),
                shape = RoundedCornerShape(24.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Burgundy,
                    disabledContainerColor = Burgundy.copy(alpha = .5f),
                ),
            ) {
                Text("بگو", color = Color.White, fontSize = 31.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}
