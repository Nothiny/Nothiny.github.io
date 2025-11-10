/*!
 * Clean Blog v1.0.0 (http://startbootstrap.com)
 * Copyright 2015 Start Bootstrap
 * Licensed under Apache 2.0 (https://github.com/IronSummitMedia/startbootstrap/blob/gh-pages/LICENSE)
 */

 /*!
 * Hux Blog v1.6.0 (http://startbootstrap.com)
 * Copyright 2016 @huxpro
 * Licensed under Apache 2.0 
 */

// Tooltip Init
// Unuse by Hux since V1.6: Titles now display by default so there is no need for tooltip
// $(function() {
//     $("[data-toggle='tooltip']").tooltip();
// });


// make all images responsive
/* 
 * Unuse by Hux
 * actually only Portfolio-Pages can't use it and only post-img need it.
 * so I modify the _layout/post and CSS to make post-img responsive!
 */
// $(function() {
//  $("img").addClass("img-responsive");
// });

// responsive tables
$(document).ready(function() {
    $("table").wrap("<div class='table-responsive'></div>");
    $("table").addClass("table");
});

// responsive embed videos
$(document).ready(function() {
    $('iframe[src*="youtube.com"]').wrap('<div class="embed-responsive embed-responsive-16by9"></div>');
    $('iframe[src*="youtube.com"]').addClass('embed-responsive-item');
    $('iframe[src*="vimeo.com"]').wrap('<div class="embed-responsive embed-responsive-16by9"></div>');
    $('iframe[src*="vimeo.com"]').addClass('embed-responsive-item');
});

// copy code block
$(document).ready(function() {
    function fallbackCopyText(text, onSuccess, onError) {
        var $temp = $('<textarea>').css({
            position: 'fixed',
            top: '-9999px',
            left: '-9999px',
            opacity: 0
        }).val(text).appendTo('body');
        var tempDom = $temp.get(0);
        tempDom.focus();
        tempDom.select();
        try {
            var successful = document.execCommand('copy');
            successful ? onSuccess() : onError();
        } catch (err) {
            onError();
        }
        $temp.remove();
    }

    $('.highlighter-rouge .highlight, pre.highlight').each(function() {
        var $block = $(this);
        if ($block.find('.copy-code-btn').length) {
            return;
        }

        var $button = $('<button type="button" class="copy-code-btn" aria-label="复制代码" title="复制代码">复制</button>');

        $button.on('click', function() {
            var codeText = $block.find('.rouge-code pre').text() ||
                $block.find('pre code').text() ||
                $block.find('pre').text() ||
                $block.text();

            var resetState = function() {
                $button.removeClass('is-copied is-error').text('复制');
            };

            var showSuccess = function() {
                $button.addClass('is-copied').text('已复制');
                setTimeout(resetState, 2000);
            };

            var showError = function() {
                $button.addClass('is-error').text('复制失败');
                setTimeout(resetState, 2000);
            };

            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(codeText).then(showSuccess).catch(function() {
                    fallbackCopyText(codeText, showSuccess, showError);
                });
            } else {
                fallbackCopyText(codeText, showSuccess, showError);
            }
        });

        $block.append($button);
    });
});

// Navigation Scripts to Show Header on Scroll-Up
jQuery(document).ready(function($) {
    var MQL = 1170;

    //primary navigation slide-in effect
    if ($(window).width() > MQL) {
        var headerHeight = $('.navbar-custom').height(),
            bannerHeight  = $('.intro-header .container').height();     
        $(window).on('scroll', {
                previousTop: 0
            },
            function() {
                var currentTop = $(window).scrollTop(),
                    $catalog = $('.side-catalog');

                //check if user is scrolling up by mouse or keyborad
                if (currentTop < this.previousTop) {
                    //if scrolling up...
                    if (currentTop > 0 && $('.navbar-custom').hasClass('is-fixed')) {
                        $('.navbar-custom').addClass('is-visible');
                    } else {
                        $('.navbar-custom').removeClass('is-visible is-fixed');
                    }
                } else {
                    //if scrolling down...
                    $('.navbar-custom').removeClass('is-visible');
                    if (currentTop > headerHeight && !$('.navbar-custom').hasClass('is-fixed')) $('.navbar-custom').addClass('is-fixed');
                }
                this.previousTop = currentTop;


                //adjust the appearance of side-catalog
                $catalog.show()
                if (currentTop > (bannerHeight + 41)) {
                    $catalog.addClass('fixed')
                } else {
                    $catalog.removeClass('fixed')
                }
            });
    }
});