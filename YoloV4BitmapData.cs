using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace Demo
{
    public class YoloV4BitmapData
    {
        [ColumnName("bitmap")]
        [ImageType(416, 416)]
        public Bitmap Image { get; set; }

        [ColumnName("width")]
        public float ImageWidth => Image.Width;

        [ColumnName("height")]
        public float ImageHeight => Image.Height;
    }
}
