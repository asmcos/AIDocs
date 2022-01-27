# 🚀 Use Strapi  write AIDOCS
基于 strapi 创建了一个可以在线编辑AIDocs的系统

修改了全屏编辑模式

主要修改了 3个文件

strapi 4.0.5 版本， 2022年1月27日

1. 修改编辑框高度，下部有空白问题 修改height:95%

# vim node_modules/@strapi/admin/admin/src/content-manager/components/Wysiwyg/EditorStylesContainer.js

```
.CodeMirror {
/* Set height, width, borders, and global font properties here */
font-size: ${14 / 16}rem;
height: calc(95%);
color: ${({ theme }) => theme.colors.neutral800};
direction: ltr;
font-family: --apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell,
'Open Sans', 'Helvetica Neue', sans-serif;
}
```
2.1由70%修改成 95% 全屏模式, 

#vim node_modules/@strapi/admin/admin/src/content-manager/components/Wysiwyg/EditorLayout.js

```
<Box
id="wysiwyg-expand"
background="neutral0"
hasRadius
shadow="popupShadow"
overflow="hidden"
width="95%"
height="95%"
onClick={e => e.stopPropagation()}
>
```

2.2

预览右侧滚动 原高度100%，改成90%


```
<Box position="relative" height="90%">
<PreviewWysiwyg data={previewContent} />
</Box>

```


3. 修改成滑动模式 修改高度90%，overflow：scroll
# vim node_modules/@strapi/admin/admin/src/content-manager/components/Wysiwyg/WysiwygStyles.js

```
export const EditorAndPreviewWrapper = styled.div`
position: relative;
height:90%;
overflow:scroll;
`;
```
